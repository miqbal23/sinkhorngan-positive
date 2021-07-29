import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from base import BaseGanCriterion


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

class One_Sided(nn.Module):
    def __init__(self):
        super(One_Sided, self).__init__()
        main = nn.ReLU()
        self.main = main
    
    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output

# uncomment if using global One_Sided object for both G & D
# one_sided = One_Sided()
# one_sided.apply(init_weights)

# def one_sided(_input):
#     output = F.relu(-_input)
#     output = -output.mean()
#     return output

def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    """
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    """
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'


class Critic(BaseGanCriterion):
    def __init__(self, parent, regularization=None, **kwargs):
        super().__init__(parent=parent, regularization=regularization, **kwargs)
        base = self.configs['base']
        sigma_list = self.configs['sigma_list']
        self.sigma_list = [sigma / base for sigma in sigma_list]
        self.lambda_MMD = self.configs['lambda_MMD'] if 'lambda_MMD' in self.configs else 1.0
        self.lambda_AE_X = self.configs['lambda_AE_X'] if 'lambda_AE_X' in self.configs else 8.0
        self.lambda_AE_Y = self.configs['lambda_AE_Y'] if 'lambda_AE_Y' in self.configs else 8.0
        self.lambda_rg = self.configs['lambda_rg'] if 'lambda_rg' in self.configs else 16.0

        # uncomment if using private One_Sided object for C only
        self.one_sided = One_Sided()
        self.one_sided.apply(init_weights)

    def calculate(self, z, x, y):
        # Check D with real data
        critic_real = self.model["critic"](x)

        # Check D with fake data generated by G
        gen_x = Variable(self.model["generator"](z).data)
        critic_fake = self.model["critic"](gen_x)

        # Check if multiple output (encoder-decoder)
        if len(critic_real) == 2 and len(critic_fake) == 2:
            f_enc_X_D, f_dec_X_D = critic_real
            f_enc_Y_D, f_dec_Y_D = critic_fake

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, self.sigma_list)
            mmd2_D = F.relu(mmd2_D)

            # compute rank hinge loss
            # print('f_enc_X_D:', f_enc_X_D.size())
            # print('f_enc_Y_D:', f_enc_Y_D.size())
            # one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))      # for using global one_sided
            one_side_errD = self.one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))   # for using critic-only one_sided

            # compute L2-loss of AE
            L2_AE_X_D = match(x.view(x.shape[0], -1), f_dec_X_D, 'L2')
            L2_AE_Y_D = match(y.view(y.shape[0], -1), f_dec_Y_D, 'L2')

            loss = torch.sqrt(
                mmd2_D) + self.lambda_rg * one_side_errD - self.lambda_AE_X * L2_AE_X_D - self.lambda_AE_Y * L2_AE_Y_D
        else:
            # compute biased MMD2 and use ReLU to prevent negative value
            loss = mix_rbf_mmd2(critic_real, critic_fake, self.sigma_list)
            loss = F.relu(loss)
            loss = torch.sqrt(loss)

        # Return dict for regularization parameters
        reg_params = dict(
            network=self.model['critic'],
            real_data=x,
            fake_data=gen_x,
            critic_real=critic_real,
            critic_fake=critic_fake,
        )
        return loss, reg_params


class Generator(BaseGanCriterion):
    def __init__(self, parent, regularization=None, **kwargs):
        super().__init__(parent=parent, regularization=regularization, **kwargs)
        base = self.configs['base']
        sigma_list = self.configs['sigma_list']
        self.sigma_list = [sigma / base for sigma in sigma_list]
        self.lambda_MMD = self.configs['lambda_MMD'] if 'lambda_MMD' in self.configs else 1.0
        self.lambda_AE_X = self.configs['lambda_AE_X'] if 'lambda_AE_X' in self.configs else 8.0
        self.lambda_AE_Y = self.configs['lambda_AE_Y'] if 'lambda_AE_Y' in self.configs else 8.0
        self.lambda_rg = self.configs['lambda_rg'] if 'lambda_rg' in self.configs else 16.0

        # uncomment if using private One_Sided object for G only
        self.one_sided = One_Sided()
        self.one_sided.apply(init_weights)

    def calculate(self, z, x, y):
        # Check D with real data
        critic_real = self.model["critic"](x)

        # Check G net
        gen_x = self.model["generator"](z)
        critic_fake = self.model["critic"](gen_x)

        # Check if multiple output (encoder-decoder)
        if len(critic_real) == 2 and len(critic_fake) == 2:
            f_enc_X, f_dec_X = critic_real
            f_enc_Y, f_dec_Y = critic_fake

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, self.sigma_list)
            mmd2_G = F.relu(mmd2_G)

            # compute rank hinge loss
            # print('f_enc_X_D:', f_enc_X_D.size())
            # print('f_enc_Y_D:', f_enc_Y_D.size())
            # compute rank hinge loss
            # one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))      # for using global one_sided
            one_side_errG = self.one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))   # for using gen-specific one_sided

            # compute L2-loss of AE
            loss = torch.sqrt(mmd2_G) + self.lambda_rg * one_side_errG

        else:
            # compute biased MMD2 and use ReLU to prevent negative value
            loss = mix_rbf_mmd2(critic_real, critic_fake, self.sigma_list)
            loss = F.relu(loss)
            # Calculate criterion
            loss = torch.sqrt(loss)

        # Return dict for regularization parameters
        reg_params = dict(
            network=self.model['critic'],
            real_data=x,
            fake_data=gen_x,
            critic_real=critic_real,
            critic_fake=critic_fake,
        )
        return loss, reg_params


# !/usr/bin/env python
# encoding: utf-8

import torch

min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal ** 2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY ** 2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    var_est = (
            2.0 / (m ** 2 * (m - 1.0) ** 2) * (
            2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
            - (4.0 * m - 6.0) / (m ** 3 * (m - 1.0) ** 3) * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 4.0 * (m - 2.0) / (m ** 3 * (m - 1.0) ** 2) * (
                    K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
            - 4.0 * (m - 3.0) / (m ** 3 * (m - 1.0) ** 2) * (K_XY_2_sum) - (8 * m - 12) / (
                    m ** 5 * (m - 1)) * K_XY_sum ** 2
            + 8.0 / (m ** 3 * (m - 1.0)) * (
                    1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
                    - Kt_XX_sums.dot(K_XY_sums_1)
                    - Kt_YY_sums.dot(K_XY_sums_0))
    )
    return mmd2, var_est

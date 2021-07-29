import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.trainer import UpdaterBase


class Sinkhorn_EOT(UpdaterBase):

    def __init__(self, params, network, pre_train=None):
        super(Sinkhorn_EOT, self).__init__(params, network, pre_train)

        self.one = self.FloatTensor([1])
        self.mone = self.one * -1

        self.epsilon = self.params["Sinkhorn"]["epsilon"]
        self.niter_sink = self.params["Sinkhorn"]["niter_sink"]
        self.margin = self.params["Sinkhorn"]["margin"]
        self.cutoff = 'relu'
        self.ratio = self.FloatTensor([1.])
        reduction = "sum"

        # Set optimizer
        for name in self.models.keys():
            self.models.set_optim(net=name, optimizer=torch.optim.Adam,
                                  lr=self.params["optimizer"]["lr_" + name],
                                  betas=self.params["optimizer"]["betas"])

        self.sinkhorn_loss = SinkhornDistance(eps=self.epsilon, max_iter=self.niter_sink, reduction=reduction)
        # self.sinkhorn_loss = SinkhornDistance_POT(eps=self.epsilon, max_iter=self.niter_sink, reduction=reduction)

        # add fixed sample for testing
        self.fixed_samples = {
            "z_input": self.FloatTensor(
                np.random.normal(0, 1, (self.params["train"]["generated_size"], self.params["network"]["z_dim"])))
        }

    def update_parameters_discriminator(self, z, x, y):

        ## split x and z
        assert x.shape[0] % 2 == 0 and z.shape[0] % 2 == 0, "Make sure input batch is devided by 2"
        _x, _xv = torch.split(x, x.shape[0] // 2, dim=0)
        _z, _zv = torch.split(z, z.shape[0] // 2, dim=0)

        # Check D with real data
        d_real = self.models["Dis"](_x)
        d_realv = self.models["Dis"](_xv)

        # Check D with fake data generated by G
        gen_x = self.models["Gen"](_z)
        d_fake = self.models["Dis"](gen_x)

        # calculate criterion
        loss_real_fake = self.sinkhorn_loss(d_real, d_fake)[0]
        loss_real_realv = self.sinkhorn_loss(d_real, d_realv)[0]
        loss = (loss_real_realv - loss_real_fake) + self.margin

        if self.cutoff is "leaky":
            loss = F.leaky_relu(loss)
        elif self.cutoff is "relu":
            loss = F.relu(loss)
        else:
            pass

        loss.backward()
        return loss

    def update_parameters_generator(self, z, x, y):

        ## split x and z
        assert x.shape[0] % 2 == 0 and z.shape[0] % 2 == 0, "Make sure input batch is devided by 2"
        _x, _xv = torch.split(x, x.shape[0] // 2, dim=0)
        _z, _zv = torch.split(z, z.shape[0] // 2, dim=0)

        # Check D with real data
        d_real = self.models["Dis"](_x)

        # Check G net
        gen_x = self.models["Gen"](_z)
        d_fake = self.models["Dis"](gen_x)

        # calculate criterion, all return [cost, pi, C] we need only cost
        loss = self.sinkhorn_loss(d_real, d_fake)[0]
        loss.backward()

        return loss


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

        def _cost_matrix(x, y, p=2):
            "Returns the matrix of $|x_i-y_j|^p$."
            x_col = x.unsqueeze(-2)
            y_lin = y.unsqueeze(-3)
            C = (torch.abs(x_col - y_lin)) ** p
            C = torch.sum(C, -1)
            return C

        # The Sinkhorn algorithm takes as input three variables :
        C = _cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]

        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().type(self.FloatTensor)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().type(self.FloatTensor)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-9
        with torch.no_grad():
            # Sinkhorn iterations
            for _ in range(self.max_iter):
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(M(u, v), dim=-1)) + u
                v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1)) + v
                err = (u - u1).abs().sum(-1).mean()

                actual_nits += 1
                if err.item() < thresh:
                    break

            U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(M(U, V))

        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class SinkhornDistance_POT(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance_POT, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):

        def _cost_matrix(x, y, p=2):
            "Returns the matrix of $|x_i-y_j|^p$."
            x_col = x.unsqueeze(-2)
            y_lin = y.unsqueeze(-3)
            C = (torch.abs(x_col - y_lin)) ** p
            C = torch.sum(C, -1)
            return C

        # The Sinkhorn algorithm takes as input three variables :
        C = _cost_matrix(x, y)  # Wasserstein cost function
        M = C.detach().cpu().numpy()
        pi = ot.sinkhorn([], [], M, 1, method='sinkhorn_stabilized')
        cost = torch.sum(self.FloatTensor(pi) * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

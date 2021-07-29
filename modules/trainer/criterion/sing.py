import torch
from torch import nn
from geomloss import SamplesLoss
import math

from base import BaseGanCriterion


# Adapted from https://github.com/shenzebang/Sinkhorn_Natural_Gradient/blob/main/Sinkhorn_GAN_SiNG_JKO.py

class Critic(BaseGanCriterion):
    def __init__(self, parent, regularization=None, **kwargs):
        super().__init__(parent=parent, regularization=regularization, **kwargs)

        self.sinkhorn_divergence_obj = SamplesLoss(blur=math.sqrt(self.configs['ent_reg_loss']),
                                                   backend="tensorized",
                                                   p=self.configs['p'],
                                                   scaling=self.configs['scaling'])
        self.sinkhorn_divergence_con = SamplesLoss(blur=math.sqrt(self.configs['ent_reg_cons']),
                                                   backend="tensorized",
                                                   p=self.configs['p'],
                                                   scaling=self.configs['scaling'])

    def calculate(self, z, x, y):
        loss_scalar = {}
        gen_x = self.model['generator'](z).detach()
        weight_half = (torch.ones(self.batch_size // 2) / (self.batch_size / 2)).to(self.device)
        # --- train the encoder of ground cost:
        # ---   optimizing on the ground space endowed with cost \|φ(x) - φ(y)\|^2 is equivalent to
        # ---   optimizing on the feature space with cost \|x - y\|^2.
        critic_real1, critic_real2 = self.model['critic'](x).chunk(2, dim=0)
        critic_fake1, critic_fake2 = self.model['critic'](gen_x).chunk(2, dim=0)
        # 1
        negative_loss = - self.sinkhorn_divergence_obj(weight_half, critic_real1, weight_half, critic_fake1)
        loss_scalar['negative_loss1'] = negative_loss.item()
        # 2
        negative_loss = - self.sinkhorn_divergence_obj(weight_half, critic_real2, weight_half,
                                                       critic_fake2) + negative_loss
        loss_scalar['negative_loss2'] = negative_loss.item()
        # 3
        negative_loss = - self.sinkhorn_divergence_obj(weight_half, critic_real1, weight_half,
                                                       critic_fake2) + negative_loss
        loss_scalar['negative_loss3'] = negative_loss.item()
        # 4
        negative_loss = - self.sinkhorn_divergence_obj(weight_half, critic_real2, weight_half,
                                                       critic_fake1) + negative_loss
        loss_scalar['negative_loss4'] = negative_loss.item()
        # 5
        negative_loss = self.sinkhorn_divergence_obj(weight_half, critic_real1, weight_half,
                                                     critic_real2) * 2 + negative_loss
        loss_scalar['negative_loss5'] = negative_loss.item()
        # 6 or total
        negative_loss = self.sinkhorn_divergence_obj(weight_half, critic_fake1, weight_half,
                                                     critic_fake2) * 2 + negative_loss
        # critic_real = self.model['critic'](x)
        # critic_fake = self.model['critic'](self.model['generator'](z))
        # negative_loss = -sinkhorn_divergence(μ_weight, critic_fake, x_real_weight, critic_real)
        loss = negative_loss
        torch.cuda.empty_cache()

        # Return dict for regularization parameters
        reg_params = dict(
            network=self.model['critic'],
            real_data=x,
            fake_data=gen_x,
            critic_real=[critic_real1, critic_real1],
            critic_fake=[critic_fake1, critic_fake2],
        )
        self.logger(Scalar=loss_scalar)
        return loss, reg_params


class Generator(BaseGanCriterion):
    def __init__(self, parent, regularization=None, **kwargs):
        super().__init__(parent=parent, regularization=regularization, **kwargs)
        self.sinkhorn_divergence_obj = SamplesLoss(blur=math.sqrt(self.configs['ent_reg_loss']),
                                                   backend="tensorized",
                                                   p=self.configs['p'],
                                                   scaling=self.configs['scaling'])
        self.sinkhorn_divergence_con = SamplesLoss(blur=math.sqrt(self.configs['ent_reg_cons']),
                                                   backend="tensorized",
                                                   p=self.configs['p'],
                                                   scaling=self.configs['scaling'])

    def calculate(self, z, x, y):
        weight = (torch.ones(self.batch_size) / self.batch_size).to(self.device)
        # train the decoder with SiNG-JKO
        with torch.autograd.no_grad():
            gen_x = self.model['generator'](z)
            critic_fake_before = self.model['critic'](gen_x)
            critic_real = self.model['critic'](x)

        # temporarily freeze the parameters of the encoder to reduce computation
        for param in self.model['critic'].parameters():
            param.requires_grad = False

        for i_jko in range(self.configs['jko_steps']):
            self.model.optim('generator').zero_grad()
            critic_fake = self.model['critic'](self.model['generator'](z))
            loss_jko = self.sinkhorn_divergence_obj(weight, critic_fake, weight, critic_real) \
                       + self.configs['eta'] * self.sinkhorn_divergence_con(weight, critic_fake, weight,
                                                                            critic_fake_before)
            loss_jko.backward()
            self.model.optim('generator').step()

        # unfreeze the parameters of the encoder
        for param in self.model['critic'].parameters():
            param.requires_grad = True

        # evaluate the sinkhorn divergence after the JKO update
        with torch.autograd.no_grad():
            critic_fake_after = self.model['critic'](self.model['generator'](z))
            s_delta = self.sinkhorn_divergence_con(weight, critic_fake_before, weight,
                                                   critic_fake_after)

        # evaluate the objective value after the JKO update
        with torch.autograd.no_grad():
            critic_fake = self.model['critic'](self.model['generator'](z))
            loss = self.sinkhorn_divergence_obj(weight, critic_fake, weight, critic_real)

        # Return dict for regularization parameters
        reg_params = dict(
            network=self.model['generator'],
            real_data=x,
            fake_data=gen_x,
            critic_real=critic_real,
            critic_fake=critic_fake,
        )
        loss_scalar = {
            "generator_loss_delta": s_delta.item()
        }
        self.logger(Scalar=loss_scalar)

        # manually increment generator counter
        self.inc_counter('generator')
        return loss, reg_params

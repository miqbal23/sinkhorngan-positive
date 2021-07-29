import torch
from torch import autograd
from torch.autograd import Variable

from base.container import Container


class WasserstainDiv(Container):
    def __call__(self, real_data, fake_data, critic_real, critic_fake, **kwargs):
        norm_power = self.configs['norm_power']
        div_coeff = self.configs['div_coeff']

        # calculate Wasserstein divergence
        real_grad_out = Variable(self.Tensor.Float(real_data.size(0), 1).fill_(1.0), requires_grad=False)
        real_grad = autograd.grad(
            critic_real, real_data, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (norm_power / 2)

        fake_grad_out = Variable(self.Tensor.Float(fake_data.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(
            critic_fake, fake_data, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (norm_power / 2)

        return torch.mean(real_grad_norm + fake_grad_norm) * div_coeff

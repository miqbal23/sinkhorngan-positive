import torch
from torch import autograd

from base.container import Container


class GradientPenalty(Container):

    def __call__(self, network, real_data, fake_data, **kwargs):
        # Random weight term for interpolation between real and fake samples
        alpha_shape = tuple([s if i < 0 else 1 for i, s in enumerate(real_data.shape)])
        alpha = torch.rand(*alpha_shape).type(real_data.type())

        # Get random interpolation between real and fake samples
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        d_interpolates = network(interpolates.requires_grad_(True))
        fake = torch.ones(*d_interpolates.shape).type(real_data.type())
        # Get gradient w.r.t. interpolates
        grads = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grads = grads.view(grads.size(0), -1)
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty * self.configs['lambda']

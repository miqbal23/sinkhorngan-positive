from base import BaseUpdater


# Adapted from https://github.com/shenzebang/Sinkhorn_Natural_Gradient/blob/main/Sinkhorn_GAN_SiNG_JKO.py

class GAN(BaseUpdater):
    n_critic = 5

    def update_parameters(self, idx, x, y):
        loss_generator = None
        loss_critic = None
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)

        z = self.Tensor.randn(x.shape[0], self.configs['z_dim'], mean=z_mean, std=z_std)

        self.model.zero_grad("critic")
        loss_critic = self.trainer.critic['criterion'](z=z, x=x, y=y)
        loss_critic.backward()
        self.model.step("critic")

        if self.get_counter('critic') % self.n_critic == 0:
            # all common process like zero_grad, backward and step performs in the generator criterion
            loss_generator = self.trainer.generator['criterion'](z=z, x=x, y=y)

        logs_scalar = {
            'generator_loss_total': loss_generator.item() if loss_generator else None,
            'critic_loss_total': loss_critic.item()
        }
        self.logger(Scalar=logs_scalar)

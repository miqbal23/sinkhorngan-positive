from base import BaseUpdater


class GAN(BaseUpdater):
    def update_parameters(self, idx, x, y):
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)

        z = self.Tensor.randn(x.shape[0], self.configs['z_dim'], mean=z_mean, std=z_std)

        self.model.zero_grad("generator")
        loss_generator = self.trainer.generator['criterion'](z=z, x=x, y=y)
        loss_generator.backward()
        self.model.step("generator")

        self.model.zero_grad("critic")
        loss_critic = self.trainer.critic['criterion'](z=z, x=x, y=y)
        loss_critic.backward()
        self.model.step("critic")

        logs_scalar = {
            'generator_loss_total': loss_generator.item() if loss_generator else None,
            'critic_loss_total': loss_critic.item()
        }
        self.logger(Scalar=logs_scalar)

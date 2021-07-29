from base import BaseUpdater


class GAN(BaseUpdater):
    n_critic = 5

    def update_parameters(self, idx, x, y):
        if self.get_counter('generator') < 25 or self.get_counter('generator') % 500 == 0:
            self.n_critic = 100
        else:
            self.n_critic = 5

        loss_generator = None
        loss_critic = None

        try:
            for p in self.model["critic"].encoder.parameters():
                p.data.clamp_(-0.01, 0.01)
        except:
            for p in self.model["critic"].parameters():
                p.data.clamp_(-0.01, 0.01)

        # Init noise
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)
        z = self.Tensor.randn(x.shape[0], self.configs['z_dim'], mean=z_mean, std=z_std)
        
        # Set critic grad true
        for p in self.model["critic"].parameters():
            p.requires_grad = True

        self.model.zero_grad("critic")
        loss_critic = self.trainer.critic['criterion'](z=z, x=x, y=y)
        loss_critic.backward(self.Tensor.MoneFloat)
        self.model.step("critic")

        self.model.zero_grad("generator")
        if self.get_counter('critic') % self.n_critic == 0:
            # Set critic grad false
            for p in self.model["critic"].parameters():
                p.requires_grad = False
            loss_generator = self.trainer.generator['criterion'](z=z, x=x, y=y)
            loss_generator.backward(self.Tensor.OneFloat)
            self.model.step("generator")

        logs_scalar = {
            'generator_loss_total': loss_generator.item() if loss_generator else None,
            'critic_loss_total': loss_critic.item()
        }
        self.logger(Scalar=logs_scalar)

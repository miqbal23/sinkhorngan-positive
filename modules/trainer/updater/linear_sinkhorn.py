from base import BaseUpdater


class GAN(BaseUpdater):
    #     n_critic = 1

    def update_parameters(self, idx, x, y):
        loss_generator = None
        loss_critic = None
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)

        # set n_critic
        if self.dataloader['main'].name == 'MnistDataLoader':
            self.n_critic = 5
        else:
            if self.epoch < 25 or self.epoch % 500 == 0:
                self.n_critic = 10
            else:
                self.n_critic = 1

        z = self.Tensor.randn(x.shape[0], self.configs['z_dim'], mean=z_mean, std=z_std)

        # set require grad to true
        for p in self.model["critic"].parameters():
            p.requires_grad = True

        # clamp
        for p in self.model["critic"].parameters():
            p.data.clamp_(-0.01, 0.01)

        self.model.zero_grad("critic")
        loss_critic = self.trainer.critic['criterion'](z=z, x=x, y=y)
        loss_critic.backward()
        self.model.step("critic")

        self.model.zero_grad("generator")
        if self.get_counter('critic') % self.n_critic == 0:
            # set require grad to false
            for p in self.model["critic"].parameters():
                p.requires_grad = False
            loss_generator = self.trainer.generator['criterion'](z=z, x=x, y=y)
            loss_generator.backward()
            self.model.step("generator")

        logs_scalar = {
            'generator_loss_total': loss_generator.item() if loss_generator else None,
            'critic_loss_total': loss_critic.item()
        }
        self.logger(Scalar=logs_scalar)

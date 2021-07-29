from abc import abstractmethod

from torch.autograd import Variable
from tqdm import tqdm

from base.container import Container


class BaseUpdater(Container):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

    def an_epoch(self):
        self.model.train()
        data_loader = self.dataloader['main']()
        for idx, (x, y) in enumerate(
                tqdm(data_loader, desc="Epoch {epoch}/{epochs}".format(epoch=self.epoch, epochs=self.len_epoch))):
            self.inc_global_step()

            x = Variable(x.to(self.device), requires_grad=self.dataloader['main'].requires_grad)
            y = y.to(self.device)

            # Update parameters
            self.update_parameters(idx, x, y)

            # Eval if step
            self.evaluator()

            # Save model and log if step
            self.trainer.save_model()
            self.trainer.save_logs()

            # for handle multiple run
            self.inc_by_float()

    @abstractmethod
    def update_parameters(self, idx, x, y):
        pass

from tensorboardX import SummaryWriter

import modules
from base.container import Container
from utils import print_wline


class Logger(Container):
    """
    Aggregator for multiple Logger
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, name='logger', **kwargs)
        self.set_writer(self.paths.sess)
        self.set_modules(module=modules.logger)

    def __call__(self, **kwargs):
        for name, list_values in kwargs.items():
            if name in self.modules:
                self.modules[name].updates(**list_values)

    def flush(self, global_step=None, run_per: tuple = None):
        global_step = self.model.global_step if global_step is None else global_step
        for name, item in self.modules.items():
            item.flush(global_step=global_step, run_per=run_per)

    def warning(self, text):
        print(text)

    def set_writer(self, path):
        print_wline("Script for Check log")
        print("tensorboard --logdir {} --bind_all".format(self.paths.sess))
        self.set_params(writer=SummaryWriter(path))

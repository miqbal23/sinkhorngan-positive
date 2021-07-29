import torch
from collections import OrderedDict

class GlobalProperties:
    """
    Global Properties for hint and shortcut
    """

    def __init__(self, parent: object):
        self.parent = parent

        # Get root session, it will return None
        self.root = self
        while self.root.parent is not None:
            self.root = self.root.parent

    # Constructor properties -----------------------------------------------

    @property
    def model(self):
        return self.root.modules['model']

    @property
    def dataloader(self):
        return self.root.modules['dataloader']

    @property
    def logger(self):
        return self.root.modules['logger']

    @property
    def evaluator(self):
        return self.root.modules['evaluator']

    @property
    def trainer(self):
        return self.root.modules['trainer']

    @property
    def writer(self):
        return self.root.modules['logger'].params['writer']

    # Counter Properties -----------------------------------------------
    @property
    def counter(self):
        if 'counter' not in self.root.params:
            self.root.set_params(counter=OrderedDict(step=0, epoch=0))
        return self.root.params['counter']

    def set_counter(self, name, value=0):
        self.counter[name] = value
        if name == 'step' and self.counter['step'] % self.len_step == 0:
            self.counter['epoch'] = self.counter['step'] // self.len_step

    def get_counter(self, name='step'):
        return self.counter[name]

    @property
    def epoch(self):
        return self.get_counter('epoch')

    @property
    def global_step(self):
        return self.get_counter('step')

    def inc_counter(self, name='step', value=1):
        self.set_counter(name, int(self.counter[name]) + value)

    def inc_epoch(self, n=1):
        self.inc_counter('epoch', n)

    def inc_global_step(self, n: int = 1):
        self.inc_counter('step', n)

    def inc_by_float(self):
        for k in self.counter.keys():
            if k not in ['epoch', 'step']:
                self.counter[k] = self.counter[k] + 1e-10

    # len_of Properties -----------------------------------------------
    @property
    def len_of(self):
        return self.root.params['len_of']

    @property
    def batch_size(self):
        return self.len_of['batch_size']

    @property
    def len_epoch(self):
        return self.len_of['epoch']

    @property
    def len_step(self):
        return self.len_of['step']

    # Session Properties -----------------------------------------------
    @property
    def device(self):
        return self.root.params['device']

    @property
    def device_ids(self):
        return self.root.params['device_ids']

    @property
    def paths(self):
        return self.root.params["paths"]

    @property
    def Tensor(self):
        return self.root.params['tensor']

    def check_readiness(self, run_per: tuple):
        n, per = run_per
        assert per in self.counter, "{} not in counter session".format(per)
        if self.get_counter(per) == 0:
            return False
        if per == 'epoch':
            return (int(self.get_counter('step')) % self.len_step == 0) and (int(self.get_counter('epoch')) % n == 0)
        else:
            return self.get_counter(per) % n == 0


class Tensor(object):
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    @property
    def Float(self):
        return torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor

    @property
    def Long(self):
        return torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor

    @property
    def Bool(self):
        return torch.cuda.BoolTensor if self.device.type == 'cuda' else torch.BoolTensor

    @property
    def Char(self):
        return torch.cuda.CharTensor if self.device.type == 'cuda' else torch.CharTensor

    @property
    def OneFloat(self):
        return self.Float([1])

    @property
    def MoneFloat(self):
        return self.OneFloat * -1

    @property
    def Byte(self):
        return torch.cuda.ByteTensor if self.device.type == 'cuda' else torch.ByteTensor

    @property
    def Double(self):
        return torch.cuda.DoubleTensor if self.device.type == 'cuda' else torch.DoubleTensor

    @property
    def Short(self):
        return torch.cuda.ShortTensor if self.device.type == 'cuda' else torch.ShortTensor

    def randn(self, *size, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=tuple(size), device=self.device)

import modules
from base.container import Container


class LossRegularization(Container):
    """
    Aggregator for regularization in loss function
    """
    def __init__(self, parent, name=None, **kwargs):
        super(LossRegularization, self).__init__(parent=parent, name=name, **kwargs)
        self.set_modules(module=modules.regularization.loss)

    def __call__(self, **kwargs):
        loss = 0
        for name, module in self.modules.items():
            loss += module(**kwargs)
        return loss


class NetworkRegularization(Container):
    """
    Aggregator for regularization in Network
    """
    def __init__(self, parent, network, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self.network = network
        self.set_modules(module=modules.regularization.network)

    def __call__(self, **kwargs):
        for name, module in self.modules.items():
            module(network=self.network, **kwargs)

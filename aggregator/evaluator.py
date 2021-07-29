import modules
from base.container import Container


class Evaluator(Container):
    """
    Aggregator for multiple evaluator
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, name='evaluator', **kwargs)
        self.set_modules(module=modules.evaluator)

    def __call__(self, **kwargs):
        """
       Caller function for run each evaluator
        Args:
            **kwargs:
        """
        for name, module in self.modules.items():
            module(*kwargs)

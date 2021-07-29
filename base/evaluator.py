from abc import abstractmethod

from base.container import Container
import time

class BaseEvaluator(Container):
    """
    Base of Evaluator
    """
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

    @abstractmethod
    def eval(self, step, tag=None, run_per: tuple = None, **kwargs):
        """
        Methods for evaluation and write it on writer
        Args:
            step: step number
            tag: rename the default tag
            run_per: a tuple of run_per
            **kwargs:

        Returns:
        """
        pass

    def __call__(self, **kwargs):
        """
        Run eval with checking the readiness of evaluator
        Args:
            **kwargs: argument for eval method
        """
        if self.check_readiness(self.run_per):
            per = self.run_per[1]
            self.eval(step=self.get_counter(per), run_per=self.run_per, **kwargs)

    # @abstractmethod
    # def flush(self, step, tag, data, walltime=time.time()):
    #     pass

    @property
    def run_per(self):
        return self.configs["run_per"]

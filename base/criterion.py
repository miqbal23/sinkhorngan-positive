import os
from abc import abstractmethod

from base.container import Container


class BaseGanCriterion(Container):
    """
    Base Criterion for training gan
    """
    def __init__(self, parent, regularization=None, **kwargs):
        file_name = os.path.basename(__file__).split('.')[0]
        name = "{}-{}".format(file_name, self.__class__.__name__)
        super().__init__(parent=parent, name=name, **kwargs)
        self.regularization = regularization

    def __call__(self, *args, **kwargs):
        """
        caller function for calculating the loss function
        Args:
            *args:
            **kwargs:

        Returns:
            loss: back propagated loss

        """
        loss = self.calculate(*args, **kwargs)
        # if get multiple output
        if isinstance(loss, tuple):
            assert len(loss) == 2, "Please only return 2 parameter on loss, for loss and regularize(dict)"
            if self.regularization:
                loss = loss[0] + self.regularization(**loss[1])
            else:
                loss = loss[0]
        return loss

    @abstractmethod
    def calculate(self, *args, **kwargs):
        """
        method for calculate loss
        :param args:
        :param kwargs:
        :return: criterion, and regularize : dict
        """
        pass

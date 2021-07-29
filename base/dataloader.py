from abc import abstractmethod

from torch.utils.data import DataLoader

from base.container import Container

ARGS_LIST = {
    'dataloader': ('batch_size', 'shuffle', 'sampler',
                   'batch_sampler', 'num_workers', 'collate_fn',
                   'pin_memory', 'drop_last', 'timeout',
                   'worker_init_fn', 'multiprocessing_context')
}


class BaseDataLoader(Container):
    """
    Base for DataLoader
    """
    def __init__(self, parent, **kwargs):
        name = self.__class__.__name__
        super().__init__(parent=parent, name=name, **kwargs)

    def __call__(self, **kwargs):
        """
        caller function for get dataloader object
        Args:
            **kwargs (object): DataLoader argument, will update the default one

        """
        configs = self.configs.copy()
        configs.update(kwargs)
        configs = self.get_dataloader_args(configs)
        return DataLoader(dataset=self.dataset, **configs)

    @abstractmethod
    def dataset(self):
        """
        Properties for dataset
        Returns:
           an instance of torch.utils.data.Dataset
        """
        pass

    @property
    def is_train(self):
        return self.configs.get('is_train', True)

    @property
    def is_download(self):
        return self.configs.get('is_download', True)

    @property
    def size(self):
        return self.configs.get('size', None)

    @property
    def requires_grad(self):
        return self.configs.get('requires_grad', False)

    @property
    def dataloader_args(self):
        return self.get_dataloader_args(self.configs)

    def get_dataloader_args(self, configs):
        return {key: configs[key] for key in configs if key in ARGS_LIST['dataloader']}

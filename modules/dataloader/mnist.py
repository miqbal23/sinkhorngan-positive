from torchvision import datasets, transforms

from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):

    @property
    def dataset(self):
        _transforms = [
            transforms.Resize(tuple(self.configs['size'][1:])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        return datasets.MNIST(
            root=self.paths.dataset,
            train=self.is_train,
            download=self.is_download,
            transform=transforms.Compose(_transforms))

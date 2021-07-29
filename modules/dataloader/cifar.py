from torchvision import datasets, transforms

from base import BaseDataLoader


class Cifar10DataLoader(BaseDataLoader):

    @property
    def dataset(self):
        _transforms = [
            transforms.Resize(tuple(self.size[1:])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        return datasets.CIFAR10(
            root=self.paths.dataset,
            train=self.is_train,
            download=self.is_download,
            transform=transforms.Compose(_transforms))


class Cifar100DataLoader(BaseDataLoader):

    @property
    def dataset(self):
        _transforms = [
            transforms.Resize(tuple(self.size[1:])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        return datasets.CIFAR100(
            root=self.paths.dataset,
            train=self.is_train,
            download=self.is_download,
            transform=transforms.Compose(_transforms))

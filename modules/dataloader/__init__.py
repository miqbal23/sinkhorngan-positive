from .cat import CatDataLoader
from .celeba import CelebADataLoader
from .cifar import Cifar10DataLoader, Cifar100DataLoader
from .dogvscat import DogVsCatDataLoader
from .mnist import MnistDataLoader

__all__ = (
    "CatDataLoader",
    "CelebADataLoader",
    "Cifar10DataLoader",
    "Cifar100DataLoader",
    "DogVsCatDataLoader",
    "MnistDataLoader",)

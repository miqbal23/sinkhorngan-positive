import os
from zipfile import ZipFile

from torchvision import datasets, transforms

from base import BaseDataLoader
from utils import makedir_if_not_exist


class DogVsCatDataLoader(BaseDataLoader):
    @property
    def dataset(self):
        _transforms = [
            transforms.Resize(tuple(self.size[1:])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        _is_valid_file = None

        return DogVsCat(root=self.paths.dataset,
                        train=self.is_train,
                        transform=transforms.Compose(_transforms),
                        download=self.is_download)


class DogVsCat(datasets.ImageFolder):

    def __init__(self, root, train=True, transform=None, download=False, is_valid_file=None):
        self.root = root

        if download:
            self.download()

        self.preprocess()

        if train:
            super(DogVsCat, self).__init__(self.train_folder, transform=transform,
                                           is_valid_file=is_valid_file)
        else:
            super(DogVsCat, self).__init__(self.test_folder, transform=transform,
                                           is_valid_file=is_valid_file)

    def preprocess(self):
        if self._check_raw_exists():
            return

        # extract train zip
        def extract_zip(zip_file, dest, delete=False):
            with ZipFile(zip_file, 'r') as zipObj:
                # Extract all the contents of zip file in raw folder
                zipObj.extractall(dest)
            if delete:
                os.remove(zip_file)

        extract_zip(self.zip_data, self.raw_folder)
        extract_zip(os.path.join(self.raw_folder, 'train.zip'), self.raw_folder, True)
        extract_zip(os.path.join(self.raw_folder, 'test1.zip'), self.raw_folder, True)

        # Sparated cat and dog on train_dir
        train_cat = os.path.join(self.train_folder, 'cat')
        train_dog = os.path.join(self.train_folder, 'dog')
        makedir_if_not_exist(train_cat)
        makedir_if_not_exist(train_dog)

        for f in os.listdir(self.train_folder):
            src_f = os.path.join(self.train_folder, f)
            if os.path.isfile(src_f):
                if 'cat' in f:
                    des_f = os.path.join(train_cat, f)
                    os.rename(src_f, des_f)
                if 'dog' in f:
                    des_f = os.path.join(train_dog, f)
                    os.rename(src_f, des_f)

    def download(self):
        if self._check_raw_exists():
            return

        if self._check_zip_exists():
            return

        raise UserWarning("Download still under construction, for now put your zip file on root folder")

    def _check_raw_exists(self):
        return os.path.exists(self.raw_folder)

    def _check_zip_exists(self):
        return os.path.exists(self.zip_data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def train_folder(self):
        return os.path.join(self.raw_folder, 'train')

    @property
    def test_folder(self):
        return os.path.join(self.raw_folder, 'train')

    @property
    def zip_data(self):
        return os.path.join(self.root, "dogs-vs-cats.zip")

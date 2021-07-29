import os
import random
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
from sys import platform
from zipfile import ZipFile

import requests
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from base import BaseDataLoader
from utils import makedir_exist_ok


class CelebADataLoader(BaseDataLoader):
    @property
    def dataset(self):
        _transforms = []

        if self.is_train:
            _transforms.append(transforms.RandomHorizontalFlip())

        _transforms += [
            transforms.CenterCrop(self.configs['crop_size']),
            transforms.Resize(tuple(self.size[1:])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]

        _is_valid_file = None

        return CelebA(root=self.paths.dataset,
                      train=self.is_train,
                      transform=transforms.Compose(_transforms),
                      download=self.is_download)


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    # CelebA images and attribute label

    def __init__(self, root='', train=True, transform=None, target_transform=None, download=True, selected_attrs=None):
        """Initialize and preprocess the CelebA dataset."""

        self.root = root

        if download:
            self.download()

        self.image_dir = os.path.join(self.raw_folder, 'images')
        self.attr_path = os.path.join(self.raw_folder, "list_attr_celeba.txt")
        if selected_attrs is None:
            self.selected_attrs = ['Black_Hair',
                                   'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        else:
            self.selected_attrs = selected_attrs

        self.transform = transform
        self.train = train
        self.train_dataset = []
        self.test_dataset = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.class_to_idx[attr_name] = i
            self.idx_to_class[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.class_to_idx[attr_name]
                label.append(values[idx] == '1')

            if (i + 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        # print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.train else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return len(self.train_dataset) if self.train else len(self.test_dataset)

    @property
    def data(self):
        return self.train_dataset if self.train else self.test_dataset

    def download(self):
        """Download the CelebA data if it doesn't exist in processed_folder already."""

        if self._check_raw_exists():
            return

        if platform == 'linux' or platform == 'linux2':
            URL = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0"

            def call_wget(zip_data):
                subprocess.call('wget -N ' + URL + " -O " +
                                zip_data, shell=True)

            if not self._check_zip_exists():
                pool = ThreadPool(4)
                pool = ThreadPool(4)  # Sets the pool size to 4
                # Open the urls in their own threads
                # and return the results
                pool.map(call_wget, [self.zip_data])
                # close the pool and wait for the work to finish
                pool.close()
                pool.join()
        elif platform == 'win32':
            URL = "https://docs.google.com/uc?export=download"
            ID = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"

            def get_confirm_token(response):
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        return value
                return None

            def save_response_content(response, destination):
                total_size = int(response.headers.get('content-length', 0))
                CHUNK_SIZE = 32 * 1024

                with open(destination, "wb") as f:
                    # for chunk in response.iter_content(CHUNK_SIZE):
                    #     if chunk: # filter out keep-alive new chunks
                    #         f.write(chunk)

                    for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total_size, unit='B', unit_scale=True,
                                      desc=destination):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)

            session = requests.Session()
            response = session.get(URL, params={'id': ID}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': ID, 'confirm': token}
                response = session.get(self.URL, params=params, stream=True)

            save_response_content(response, self.zip_data)

        print("Please wait, extract file")
        with ZipFile(self.zip_data, 'r') as zipObj:
            # Extract all the contents of zip file in raw folder
            zipObj.extractall(os.path.join(self.root, self.__class__.__name__))
        print("extract done!")

        os.rename(os.path.join(self.root, self.__class__.__name__,
                               'celeba'), self.raw_folder)

    def _check_raw_exists(self):
        return makedir_exist_ok(self.raw_folder)

    def _check_zip_exists(self):
        return os.path.exists(self.zip_data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def zip_data(self):
        return os.path.join(self.root, self.__class__.__name__, "celeba.zip")

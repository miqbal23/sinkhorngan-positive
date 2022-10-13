from __future__ import print_function

import os
import os.path
import sys

import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity, makedir_exist_ok


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    training_file = 'training.pt'
    test_file = 'test.pt'

    classes = ['0 - normal', '1 - abnormal']

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, anomaly_class=None):
        self.test_percentage = 0.2
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if anomaly_class is None:
            raise RuntimeError('Please fill the anomaly_class argument' +
                               'anomaly_class=<listOfClass or integer>')

        if (isinstance(anomaly_class, list)):
            self.anomaly_class = anomaly_class
        if (isinstance(anomaly_class, int)):
            self.anomaly_class = [anomaly_class]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.preprocess_data()

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other dataloader
        # to return a PIL Image
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__,
                            'processed_with_anomaly_' + "_".join([str(i) for i in self.anomaly_class]))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def _check_integrity(self):
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile
        import os
        import shutil
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        makedir_exist_ok(self.raw_folder)

        # download files
        tgzfilename = self.url.rpartition('/')[2]
        download_url(self.url, self.raw_folder, tgzfilename, self.tgz_md5)
        # extract file
        with tarfile.open(os.path.join(self.raw_folder, tgzfilename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.raw_folder)
        # move to raw_folder
        src_dir = os.path.join(self.raw_folder, "cifar-10-batches-py")
        src_files = os.listdir(src_dir)
        for file_name in src_files:
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, self.raw_folder)

        # delete folder
        os.rmdir(src_dir)

        # delete file
        # os.remove(os.path.join(self.raw_folder, filename))

    def preprocess_data(self):

        if self._check_exists():
            return

        print('Processing...')
        makedir_exist_ok(self.processed_folder)

        def loadPickle(downloaded_list):
            data = []
            targets = []
            # now load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.raw_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    data.append(entry['data'])
                    if 'labels' in entry:
                        targets.extend(entry['labels'])
                    else:
                        targets.extend(entry['fine_labels'])

            data = np.vstack(data).reshape(-1, 3, 32, 32)
            data = data.transpose((0, 2, 3, 1))  # convert to HWC
            return data, targets

        # load all data from pickle
        tmp_train = loadPickle(self.train_list)
        tmp_test = loadPickle(self.test_list)
        tmp_data = np.append(tmp_train[0], tmp_test[0], 0)
        tmp_targets = np.append(tmp_train[1], tmp_test[1], 0)

        # move all anomaly class to test
        normal_data = []
        normal_targets = []
        abnormal_data = []
        abnormal_targets = []

        for i, l in zip(tmp_data, tmp_targets):
            if (l in self.anomaly_class):
                abnormal_data.append(i)
                abnormal_targets.append(l)
            else:
                normal_data.append(i)
                normal_targets.append(l)

        # convert all label to 0 - normal and 1 - abnormal
        normal_data = np.asarray(normal_data)
        normal_targets = np.zeros_like(normal_targets)

        abnormal_data = np.asarray(abnormal_data)
        abnormal_targets = np.ones_like(abnormal_targets)

        # Create new anomaly dataset based on the following data structure:
        # - anomaly dataset
        #   . -> train
        #        . -> normal
        #   . -> test
        #        . -> normal
        #        . -> abnormal

        test_idx = int(normal_targets.shape[0] * self.test_percentage)

        normal_data = normal_data[test_idx:, ]
        normal_targets = normal_targets[test_idx:, ]

        abnormal_data = np.append(normal_data[:test_idx, ], abnormal_data, 0)
        abnormal_targets = np.append(normal_targets[:test_idx, ], abnormal_targets, 0)

        training_set = (
            torch.from_numpy(normal_data).view(*normal_data.shape),
            torch.from_numpy(normal_targets).view(*normal_targets.shape).long()
        )
        test_set = (
            torch.from_numpy(abnormal_data).view(*abnormal_data.shape),
            torch.from_numpy(abnormal_targets).view(*abnormal_targets.shape).long()
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

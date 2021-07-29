from __future__ import print_function

import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets.utils import makedir_exist_ok


class KDD99(data.Dataset):
    """Class of KDD Cup 1999 Network Intrusion Dataset

    Parameters
    ----------
    data_dir : string
        Directory for saving downloaded dataset files
    download : boolean
        Flag to determine for downloading dataset files
    use_small_dataset : boolean
        Flag to use small version (10%) of KDD Dataset
    normal_class : list of string
        Lists of KDD classes (refer to training_attack_types file) that will be set as normal classes
    """

    classes = ['0 - normal', '1 - abnormal']
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 anomaly_class=None, normal_class=('normal'), percent10=True):

        self.anomaly_class = anomaly_class
        self.percent10 = percent10

        self.test_percentage = 0.2

        if normal_class:
            self.normal_class = normal_class
        else:
            self.normal_class = ('normal')

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        def fetch_kdd_data():
            data, target = fetch_kddcup99(data_home=self.raw_folder, percent10=self.percent10,
                                          download_if_missing=True, return_X_y=True)

            # convert categorycal to oneHotencoder
            kdd_cont1, kdd_cat, kdd_cont2 = np.hsplit(data, [1, 4])
            kdd_onehot = OneHotEncoder().fit_transform(kdd_cat).toarray()
            data = np.concatenate((kdd_cont1, kdd_cont2, kdd_onehot), axis=1)

            # convert target from binary to string
            targets = [s.decode("utf-8").replace('.', '') for s in target]
            return np.asarray(data).astype(int), np.asarray(targets)

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        tmp_data, tmp_targets = fetch_kdd_data()

        # process and save as torch files
        print('Processing...')

        normal_data = []
        normal_targets = []
        abnormal_data = []
        abnormal_targets = []

        for d, t in zip(tmp_data, tmp_targets):
            if t in self.normal_class:
                normal_data.append(d)
                normal_targets.append(t)
            else:
                abnormal_data.append(d)
                abnormal_targets.append(t)

        # convert all label to 0 - normal and 1 - abnormal
        normal_data = np.asarray(normal_data)
        normal_targets = np.zeros_like(normal_targets, dtype=int)

        abnormal_data = np.asarray(abnormal_data)
        abnormal_targets = np.ones_like(abnormal_targets, dtype=int)

        # Create new anomaly dataset based on the following data structure:
        # - anomaly dataset
        #   . -> train
        #        . -> normal
        #   . -> test
        #        . -> normal
        #        . -> abnormal

        test_idx = int(normal_targets.shape[0] * self.test_percentage)

        training_data = normal_data[test_idx:, ]
        training_targets = normal_targets[test_idx:, ]

        test_data = np.append(normal_data[:test_idx, ], abnormal_data, 0)
        test_targets = np.append(normal_targets[:test_idx, ], abnormal_targets, 0)

        training_set = (
            torch.from_numpy(training_data).view(*training_data.shape),
            torch.from_numpy(training_targets).view(*training_targets.shape).long()
        )
        test_set = (
            torch.from_numpy(test_data).view(*test_data.shape),
            torch.from_numpy(test_targets).view(*test_targets.shape).long()
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return image and its corresponding target given idx"""

        feature, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @property
    def processed_folder(self):
        if self.anomaly_class:
            return os.path.join(self.root, self.__class__.__name__,
                                'processed_with_anomaly_' + "_".join([str(i) for i in self.anomaly_class]))
        elif self.normal_class:
            return os.path.join(self.root, self.__class__.__name__,
                                'processed_with_normal_class_' + "_".join([str(i) for i in self.normal_class]))

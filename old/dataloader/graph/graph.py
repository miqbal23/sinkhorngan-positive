import json
import os
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urlparse

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils import data

from _old.dataloader.graph import utils as g_utils
from utils import makedir_exist_ok


class Graph(data.Dataset):
    """Dataset class for the CelebA dataset."""
    # CelebA images and attribute labels
    URL = ""
    training_file = 'training.pt'
    test_file = 'test.pt'
    meta_file = "meta.pt"

    def __init__(self, root="", train=True, transform=None, download=True, input_size=(16,), p=1, q=1, num_data=2000):
        """Initialize and preprocess the CelebA dataset."""

        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size[0]
        assert self.input_size > 1, "Random walk length must be > 1., please set on your params['dataloader']['input_size'] > 1"
        self.p, self.q = p, q
        self.num_data = num_data

        if download:
            self.download()

        self.load_data()

        self.class_to_idx = {}
        self.idx_to_class = {}

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.preprocess_data()

        self.data, self.data_discrete = torch.load(os.path.join(self.processed_folder, data_file))

    def preprocess_data(self):

        if self._check_exists():
            return

        print('Processing for the the first time...')
        makedir_exist_ok(self.processed_folder)

        # pre-process the graph
        graph_train, graph_test = self.preprocess()
        # create random walker for training set
        walker = g_utils.RandomWalker(adj=graph_train, input_size=self.input_size, p=self.p, q=self.q,
                                      batch_size=self.num_data)
        data_discrete = walker.walk().__next__()
        data_discrete = torch.from_numpy(data_discrete).view(*data_discrete.shape)
        data = nn.functional.one_hot(data_discrete, num_classes=self.n_node)
        training_set = (data, data_discrete)

        # create random walker for test set
        walker = g_utils.RandomWalker(adj=graph_train, input_size=self.input_size, p=self.p, q=self.q,
                                      batch_size=self.num_data)
        data_discrete = walker.walk().__next__()
        data_discrete = torch.from_numpy(data_discrete).view(*data_discrete.shape)
        data = nn.functional.one_hot(data_discrete, num_classes=self.n_node)
        test_set = (data, data_discrete)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        # save metadata
        metadata = dict(
            num_data=self.num_data,
            n_node=self.n_node,
            input_size=self.input_size,
            q=self.q, p=self.p,
            shape=data.shape)

        json.dump(metadata, open(self.meta_file, 'w'))

    def load_data(self):
        val_share = 0.1
        test_share = 0.05
        seed = 481516234

        _A_obs, _X_obs, _z_obs = g_utils.load_npz(self.npz_data)
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        lcc = g_utils.largest_connected_components(_A_obs)
        _A_obs = _A_obs[lcc, :][:, lcc]
        _N = _A_obs.shape[0]
        self.A_obs, self.n_node = _A_obs, _N

        self.train_ones, self.val_ones, self.val_zeros, self.test_ones, self.test_zeros = g_utils.train_val_test_split_adjacency(
            self.A_obs,
            val_share,
            test_share,
            seed,
            undirected=True,
            connected=True,
            asserts=True)

    def preprocess(self):

        graph_data_train = sp.coo_matrix(
            (np.ones(len(self.train_ones)), (self.train_ones[:, 0], self.train_ones[:, 1]))).tocsr()
        assert (graph_data_train.toarray() == graph_data_train.toarray().T).all()

        graph_data_test = sp.coo_matrix(
            (np.ones(len(self.test_ones)), (self.test_ones[:, 0], self.test_ones[:, 1]))).tocsr()
        assert (graph_data_test.toarray() == graph_data_test.toarray().T).all()
        return graph_data_train, graph_data_test

    def __getitem__(self, index):
        return self.data[index].type(torch.LongTensor), self.data_discrete[index].type(torch.LongTensor)

    def __len__(self):
        """Return the number of images."""
        return self.num_data

    def download(self):
        """Download the CelebA data if it doesn't exist in processed_folder already."""

        if self._check_raw_exists():
            return

        def call_wget(zip_data):
            subprocess.call('wget -N ' + self.URL + " -O " +
                            zip_data, shell=True)

        if not self._check_npz_exists():
            pool = ThreadPool(4)
            pool = ThreadPool(4)  # Sets the pool size to 4
            # Open the urls in their own threads
            # and return the results
            pool.map(call_wget, [self.npz_data])
            # close the pool and wait for the work to finish
            pool.close()
            pool.join()

    def _check_raw_exists(self):
        return makedir_exist_ok(self.raw_folder)

    def _check_npz_exists(self):
        return os.path.exists(self.npz_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @property
    def meta_file(self):
        return os.path.join(self.root, self.__class__.__name__,
                            'processed_with_rw_len{rw_len}'.format(rw_len=self.input_size), "meta.json")

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__,
                            'processed_with_rw_len{rw_len}'.format(rw_len=self.input_size))

    @property
    def npz_data(self):
        return os.path.join(self.raw_folder, os.path.basename(urlparse(self.URL).path))


class Citeseer(Graph):
    URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/citeseer.npz"


class Cora(Graph):
    URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"


class CoraML(Graph):
    URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"


class DBLP(Graph):
    URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"


class PubMed(Graph):
    URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/pubmed.npz"

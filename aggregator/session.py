import math
import os
import warnings
from collections import OrderedDict
from os.path import abspath
from shutil import copyfile

import torch
import yaml
from jinja2 import Environment

import aggregator as module_constructor
from base.container import Container
from base.properties import Tensor
from utils import print_wline


class Paths:
    def __init__(self, **kwargs):
        self._data = OrderedDict()
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
        self._data[k] = v
        setattr(self, k, self._data[k])

    def items(self):
        return self._data.items()


def create_if_not_exist(paths):
    def _create(_path, _name=''):
        if not os.path.exists(_path):
            os.makedirs(_path)
            print("[Created] {name} folder : {path}".format(name=_name, path=_path))

    if isinstance(paths, Paths):
        for name, path in paths.items():
            _create(path, name)
    else:
        _create(paths)


def load_from_yaml(path):
    assert os.path.exists(path), "Path doesn't exist !"
    assert path.split('.')[-1] in ['yaml', 'yml'], "not yaml file !"
    is_global = True
    is_hparam = False
    general_config = []
    hparams = []
    modules = []
    with open(path, 'r') as f:
        for text in f.readlines():
            text = text.replace('\n', '')
            if text.strip().startswith('#') or not text.strip():
                if text.strip().lower().startswith('# start hparam'):
                    is_hparam = True
                elif text.strip().lower().startswith('# stop hparam'):
                    is_hparam = False
                continue
            if text.startswith("---"):
                is_global = False
                is_hparam = False
                continue
            if is_hparam:
                hparams.append(text)
            if is_global:
                general_config.append(text)
            else:
                modules.append(text)
    # Load the yaml file
    general_config = yaml.safe_load("\n".join(general_config))
    hparams = yaml.safe_load("\n".join(hparams)) if len(hparams) > 0 else {}

    # render the general config to modules
    modules = Environment().from_string(str("\n".join(modules))).render(general_config)
    modules = yaml.safe_load(modules)

    return dict(configs=general_config, modules=modules, hparams=hparams)


def is_session_dir(path):
    if os.path.isfile(path):
        return False
    else:
        list_dir = os.listdir(path)
        assert 'config.yaml' in list_dir, "session dir must have a config.yaml!"
        assert 'model' in list_dir, "session dir must have model folder!"
        return True


class Session(Container):
    def __init__(self, path):
        """
        Start a session
        Args:
            path: path of session dir or yaml config
        """
        path = path.strip()
        self.is_continue = is_session_dir(path)

        # Check if load from previous session
        if self.is_continue:
            print("continue a Session from {} dir".format(abspath(path)))
            kwargs = load_from_yaml(os.path.join(path, 'config.yaml'))
        else:
            print("Try to initiate a Session from {} file".format(abspath(path)))
            kwargs = load_from_yaml(path)

        # Init Constructor
        super().__init__(parent=None, name='session', **kwargs)

        # Set Paths
        if self.is_continue:
            self.set_paths(path)
        else:
            self.set_paths()
            create_if_not_exist(self.paths)
            # copy configs to Session dir
            copyfile(path, os.path.join(self.paths.sess, 'config.yaml'))

        # Set device
        self.set_device()

        # Set Tensor
        self.set_tensor()

        # Build Module
        self.set_modules(module=module_constructor)

        # Set len_of
        self.set_len_of()

        # Set model device
        self.model.to(self.device)
        if len(self.device_ids) > 1 and self.device.type == 'cuda':
            self.model.set_parallel(device_ids=self.device_ids)

    def set_len_of(self):
        # set max step per epoch
        data_size = len(self.dataloader['main'].dataset.data)
        drop_last = self.dataloader['main'].configs['drop_last']
        len_step = data_size / self.configs['batch_size']
        _dict = OrderedDict(
            epoch=self.configs['n_epoch'],
            batch_size=self.configs['batch_size'] * self.configs['n_gpu_use'] if self.configs['n_gpu_use'] > 0 else 1,
            step=math.floor(len_step) if drop_last else math.ceil(len_step)
        )
        self.set_params(len_of=_dict)

    def train(self):
        # if continue from a session dir
        if self.is_continue:
            _models = sorted([int(f) for f in os.listdir(self.paths.model) if f.isdigit()])
            self.model.load(os.path.join(self.paths.model, str(_models[-1])))

        # Start Training
        print_wline("Start Train with {} Epoch !".format(self.len_epoch), line="=")
        self.trainer.pre()
        for _ in range(self.epoch + 1, self.len_epoch + 1):
            self.trainer()
        self.trainer.post()
        self.model.save(path=self.paths.model, global_step=-1)
        print_wline("Train with {} Epoch Done !".format(self.epoch + 1), line="=")

    def eval(self, only_last=True):
        assert self.is_continue, "Not a session dir! Please input the session dir on --config argument"
        # Reset writer to models_path
        _models = sorted([int(f) for f in os.listdir(self.paths.model) if f.isdigit()])
        assert len(_models) != 0, "There are no pretrained models in model folder!"
        if only_last:
            _models = _models[-1:]
        for m in _models:
            # load model
            self.model.load(os.path.join(self.paths.model, str(m)), verbose=False)
            self.model.to(self.device)
            if len(self.device_ids) > 1 and self.device.type == 'cuda':
                self.model.set_parallel(device_ids=self.device_ids)
            self.model.eval()
            # Start Evaluation base on configs
            for name, module in self.evaluator.modules.items():
                module.eval(step=m)

    def set_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if self.configs['n_gpu_use'] > 0 and n_gpu == 0:
            self.set_configs(n_gpu_use=n_gpu)
            warnings.warn("There\'s no GPU available on this machine, training will be performed on CPU.")
        if self.configs['n_gpu_use'] > n_gpu:
            self.set_configs(n_gpu_use=n_gpu)
            warnings.warn("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this "
                          "machine.".format(self.configs['n_gpu_use'], n_gpu))

        self.set_params(device=torch.device('cuda:0' if self.configs['n_gpu_use'] > 0 else 'cpu'))
        self.set_params(device_ids=list(range(self.configs['n_gpu_use'])))

    def set_paths(self, sess_path=None):
        root_session = self.configs['root_dir']

        def path_checker(_path):
            initial_path = _path
            _duplicate = 2
            while os.path.exists(_path):
                _path = "{path} (try {duplicate})".format(
                    path=initial_path,
                    duplicate=_duplicate)
                _duplicate += 1
            return _path

        path = "{dataset}/{trainer}/{model}/{experiment}".format(
            dataset=self.configs['dataset'],
            trainer=self.configs['trainer'],
            model=self.configs['model'],
            experiment=self.configs['note'], )

        # if any hparams add after it
        if len(self.get_hparams()) > 0:
            str_hparams = ','.join([f'{key}={value}' for key, value in self.get_hparams().items()])
            path = f"{path}/{str_hparams}"

        if sess_path is None:
            sess_path = path_checker(os.path.join(root_session, "runs", path))
        paths = Paths(root=abspath(root_session), dataset=abspath(os.path.join(root_session, "dataset")),
                      sess=abspath(sess_path),
                      model=abspath(os.path.join(sess_path, "model")))

        self.set_params(paths=paths)

        print_wline("Session Paths")
        for key, p in self.paths.items():
            print("{}\t\t: {}".format(key.title(), p))

    def set_tensor(self):
        try:
            self.set_params(tensor=Tensor(self.device))
        except KeyError:
            raise ValueError("Set device first !")

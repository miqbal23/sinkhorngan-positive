import os
import pprint
from collections import OrderedDict

import torch
from torch import optim as module_optim
from torch.optim import lr_scheduler as module_lr_scheduler

import modules
from base.container import Container
from aggregator.regularization import NetworkRegularization


class Model(Container):
    """
    Agregator for Multiple Model
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, name='model', **kwargs)
        self.set_modules(module=modules.network)

    def set_obj_module(self,
                       name: str,
                       module: any,
                       network: dict = None,
                       optim: dict = None,
                       lr_scheduler: dict = None,
                       regularization: dict = None,
                       **kwargs):

        obj_module = OrderedDict()
        _configs = OrderedDict()

        if network:
            _class = self.get_attribute(module, network['module'])
            _configs['network'] = network['module']
            _configs.update(network['configs'])
            obj_module['network'] = _class(**network['configs'])

        if optim:
            _params = [p for p in obj_module['network'].parameters() if p.requires_grad]
            _configs['optim'] = optim['module']
            _class = self.get_attribute(module_optim, optim['module'])
            obj_module['optim'] = _class(params=_params, **optim['configs'])

        if lr_scheduler:
            _class = self.get_attribute(module_lr_scheduler, lr_scheduler['module'])
            _configs['lr_scheduler'] = lr_scheduler['module']
            obj_module['lr_scheduler'] = _class(optimizer=obj_module['optim'], **lr_scheduler['configs'])

        if regularization:
            obj_module['regularization'] = NetworkRegularization(self, network=obj_module['network'], **regularization)

        # add step on params
        self.set_counter(name)
        # set attribute
        self.set_attribute(name, obj_module)
        # set config model
        self.set_configs(**{name: _configs})

    def __getitem__(self, name):
        return self.modules[name]['network']

    def optim(self, name):
        return self.modules[name]['optim']

    def get_lr(self, name):
        for param_group in self.optim(name).param_groups:
            return param_group['lr']

    def zero_grad(self, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['optim'].zero_grad()

    def step(self, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['optim'].step()
            if 'lr_scheduler' in self.modules[_name]:
                if self.check_readiness(run_per=(1, 'epoch')):
                    self.modules[_name]['lr_scheduler'].step()
                    self.writer.add_scalar("lr_{}".format(_name), self.get_lr(_name), self.epoch)
            if 'regularization' in self.modules[_name]:
                self.modules[_name]['regularization']()
            self.inc_counter(_name)

    def set_parallel(self, device_ids, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['network'] = torch.nn.DataParallel(self.modules[_name]['network'],
                                                                   device_ids=device_ids)

    def to(self, device, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['network'] = self.modules[_name]['network'].to(device)

    def cuda(self, name=None):
        assert torch.cuda.is_available(), "cuda not available !"
        for _name in self.select_item_or_all(name):
            self.modules[_name]['network'].cuda()

    def cpu(self, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['network'].cpu()

    def train(self, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['network'].train()

    def eval(self, name=None):
        for _name in self.select_item_or_all(name):
            self.modules[_name]['network'].eval()

    def save(self, path: str, global_step: int, more_info: dict = None, output_ext="pkl", name=None):
        def save_state_dict(state_dict, module_name):
            model_path = os.path.join(path, module_name + '.{ext}'.format(ext=output_ext))
            torch.save(state_dict, model_path)

        # Make dir if didn't exists
        path = os.path.join(path, str(global_step))

        if not os.path.exists(path):
            os.makedirs(path)

        # save models
        for _name in self.select_item_or_all(name):
            model = {
                'counter': self.get_counter(_name),
                'network': self.modules[_name]['network'].state_dict(),
                'optim': self.modules[_name]['optim'].state_dict(),
                'lr_scheduler': self.modules[_name]['lr_scheduler'].state_dict() if 'lr_scheduler' in self.modules[
                    _name] else {},
                'configs': self.configs[_name],
            }
            # Save Session
            save_state_dict(model, "model_{}".format(_name))

        # save info
        _info = self.configs.copy()
        _info['counter'] = OrderedDict()
        _info['counter']['step'] = self.get_counter('step')
        _info['counter']['epoch'] = self.get_counter('epoch')
        if more_info:
            _info.update(more_info)
        save_state_dict(_info, "info")

        print('[Model Saved]({})!'.format(path))

    def load(self, path: str, output_ext="pkl", name=None, verbose=True):

        def load_dict(module_name):
            model_path = os.path.join(path, module_name + '.{ext}'.format(ext=output_ext))
            return torch.load(model_path)

        def _check_model(src_info, des_info, _module, is_must=False):
            if _module not in src_info or _module not in des_info:
                return False
            if src_info[_module] != des_info[_module]:
                print_info = "{module} not same!, expected {expect} get {get}".format(
                    module=_module,
                    expect=src_info[_module]['type'],
                    get=des_info[_module]['type'])
                if is_must:
                    raise ValueError(print_info)
                else:
                    print(print_info)
                    print("so {module} not implemented".format(module=_module))
                    return False
            else:
                return True

        _configs = load_dict('info')

        print("Try to Load Model from : {path}".format(path=path))
        if verbose:
            pprint.pprint(_configs, width=1)

        # Check Models and load
        for _name in self.select_item_or_all(name):
            model = load_dict(module_name="model_{}".format(_name))
            
            # Load Network
            if 'network' in model:
                if _check_model(self.configs[_name], model['configs'], 'network', True):
                    self.modules[_name]['network'].load_state_dict(model['network'])

            # Load Optimizers
            if 'optim' in model:
                if _check_model(self.configs[_name], model['configs'], 'optim') and model['optim'] != {}:
                    self.modules[_name]['optim'].load_state_dict(model['optim'])

            # Load lr_scheduler
            if 'lr_scheduler' in model:
                if _check_model(self.configs[_name], model['configs'], 'lr_scheduler') and model['lr_scheduler'] != {}:
                    self.modules[_name]['lr_scheduler'].load_state_dict(model['lr_scheduler'])

            # Load Step
            self.set_counter(_name, model['counter'])

        # Continue Global_step and epoch
        self.set_counter('step', _configs['counter']['step'])
        self.set_counter('epoch', _configs['counter']['epoch'])
        # Set model device
        self.to(self.device)
        if len(self.device_ids) > 1 and self.device.type == 'cuda':
            self.set_parallel(device_ids=self.device_ids)
    
    # Old
    def load_old_version(self, path: str, output_ext="pkl", name=None, verbose=True):

        def load_dict(module_name):
            model_path = os.path.join(path, module_name + '.{ext}'.format(ext=output_ext))
            return torch.load(model_path)

        def check_print(src_info, des_info, _module):
            if src_info[_module] != des_info[_module]:
                print("{module} not be loaded, expected {expect} get {get}".format(
                    module=_module,
                    expect=src_info[_module]['type'],
                    get=des_info[_module]['type']))
                return False
            else:
                return True

        _configs = load_dict('info')

        print("Try to Load Model from : {path}".format(path=path))
        if verbose:
            pprint.pprint(_configs, width=1)

        # Check Models and load
        for _name in self.select_item_or_all(name):
            model = load_dict(module_name="model_{}".format(_name))

            # Load Network
            if 'network' in model:
                self.modules[_name]['network'].load_state_dict(model['network'])

            # Load Optimizers
            if 'optim' in model:
                self.modules[_name]['optim'].load_state_dict(model['optim'])

            # Load lr_scheduler
            if 'lr_scheduler' in model:
                if model['lr_scheduler'] != {}:
                    self.modules[_name]['lr_scheduler'].load_state_dict(model['lr_scheduler'])

            # Load Step
            self.set_counter(_name, model['step'])

        # Continue Global_step and epoch
        self.set_counter('step', _configs['counter']['step'])
        self.set_counter('epoch', _configs['counter']['epoch'])
        
        # Set model device
        self.to(self.device)
        if len(self.device_ids) > 1 and self.device.type == 'cuda':
            self.set_parallel(device_ids=self.device_ids)
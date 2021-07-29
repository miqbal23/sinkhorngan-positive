from collections import OrderedDict

import modules
from base.container import Container
from aggregator import LossRegularization


class Trainer(Container):
    """
    Training process
    """
    updater = None
    save_log_per = ()
    save_model_per = ()

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, name='trainer', **kwargs)

        # Set Updater
        assert 'updater' in kwargs, 'Please set updater !'
        self.set_updater(kwargs.pop('updater'))

        # build modules
        self.set_modules(module=modules.trainer)

    def set_updater(self, config):
        _class = self.get_attribute(modules.trainer.updater, config['module'])
        self.set_params(updater=_class(self, **config))
        setattr(self, 'updater', self.params['updater'])

    def set_obj_module(self,
                       name: str,
                       module: any,
                       criterion: dict = None,
                       pre: dict = None,
                       post: dict = None,
                       regularization: dict = None,
                       **kwargs):

        name = name.lower()
        obj_module = OrderedDict()

        if criterion:
            _sub_module = self.get_attribute(module, 'criterion')
            _module_name = criterion.pop('module')
            _module = self.get_attribute(_sub_module, _module_name)
            if regularization:
                regularization = LossRegularization(self, **regularization)
            obj_module['criterion'] = _module(self, regularization=regularization, **criterion)

        if pre:
            _sub_module = self.get_attribute(module, 'pre')
            _module_name = pre.pop('module')
            _module = self.get_attribute(_sub_module, _module_name)
            obj_module['pre'] = _module(self, **pre)

        if post:
            _sub_module = self.get_attribute(module, 'post')
            _module_name = post.pop('module')
            _module = self.get_attribute(_sub_module, _module_name)
            obj_module['pre'] = _module(self, **post)

        self.set_attribute(name, obj_module)

    def __call__(self):
        self.updater.an_epoch()

    def pre(self):
        for name, module in self.modules.items():
            if 'pre' in module:
                module.pre()

    def pos(self):
        for name, module in self.modules.items():
            if 'post' in module:
                module.post()

    def save_model(self):
        run_per = self.configs['save_model_per']
        if self.check_readiness(run_per=run_per):
            per = run_per[1]
            print("Try save model on {} {}".format(per, self.get_counter(per)))
            self.model.save(path=self.paths.model, global_step=self.get_counter(per))

    def save_logs(self):
        run_per = self.configs['save_log_per']
        if self.check_readiness(run_per=run_per):
            per = run_per[1]
            self.logger.flush(global_step=self.get_counter(per), run_per=self.configs['save_log_per'])

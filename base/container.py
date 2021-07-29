from abc import ABC
from collections import OrderedDict

from base.properties import GlobalProperties


class Container(GlobalProperties, ABC):
    """
    Constructor base for all purpose,
    it will prepare a container to put all info and auto get configs from it, and get the global properties
    """

    def __init__(self, parent, name=None, **kwargs):
        """
        Args:
            parent:
            name:
            **kwargs:
        each Container have 3 container (modules, configs and params)
        modules = for all object_modules or fn_modules
        configs = for all configs parameter, get from yaml file with 'configs' as name
        params = for all parameter generated when class constructed
        """
        super().__init__(parent=parent)
        # set name
        self.name = name if name else self.__class__.__name__
        # set containers
        self._modules = OrderedDict()
        self._configs = OrderedDict()
        self._params = OrderedDict()
        self._hparams = OrderedDict()

        # Set config arguments to self
        configs = kwargs.pop('configs', None)
        if configs:
            self.set_configs(**configs)

        # Set hparam
        hparams = kwargs.pop('hparams', None)
        if hparams:
            self.set_hparams(**hparams)

        # put module arguments to tmp
        if 'modules' in kwargs:
            self.__modules = kwargs.pop('modules')

    def get_modules(self):
        return self._modules

    def get_configs(self):
        return self._configs

    def get_hparams(self):
        return self._hparams

    def get_params(self):
        return self._params

    def set_modules(self, module: any):
        """
        Method to build all modules from tmp_modules
        :param module: class module to be init
        :return:
        """
        if isinstance(self.__modules, list):
            while self.__modules:
                _params = self.__modules.pop(0)
                _module_name = _params.pop('module', None)
                _name = _params.pop('name', None)
                _name = _name if _name else _module_name
                if _module_name:
                    _module = self.get_attribute(module, _module_name)
                else:
                    _module = module
                self.set_obj_module(name=_name, module=_module, **_params)
                # assert not _params, "There is unpacked parameters on {} module".format(_name)
                # del _params

            # assert not self._tmp_modules, "There is unpacked module(s) :{}".format(self._tmp_modules)
            # del self._tmp_modules
        elif isinstance(self.__modules, dict):
            for name, detail in self.__modules.items():
                _module = self.get_attribute(module, name)
                self.set_obj_module(name=name.lower(), module=_module, set_attr=False, **detail)
        else:
            raise NotImplemented()

    def set_configs(self, **kwargs):
        """
        Set all configs on self.configs
        :param kwargs: all configs dictionary
        :return:
        """
        for name, item in kwargs.items():
            # make all list to tuple
            if isinstance(item, list):
                item = tuple(item)
            self._configs[name] = item

    def set_hparams(self, **kwargs):
        """
        Set all hparams on self.hparams
        :param kwargs: all configs dictionary
        :return:
        """
        for name, item in OrderedDict(sorted(kwargs.items())).items():
            # make all list to tuple
            if isinstance(item, list):
                item = tuple(item)
            self._hparams[name] = item

    def set_params(self, **kwargs):
        """
        Set all configs on self.configs
        :param kwargs: all configs dictionary
        :return:
        """
        for name, item in kwargs.items():
            # make all list to tuple
            if isinstance(item, list):
                item = tuple(item)
            self._params[name] = item

    @staticmethod
    def get_attribute(__o: any, name: str):
        """
        get kind of attribute from a module bundle
        :param __o: bundle of module
        :param name: type of module
        :return:
        """
        if '.' in name:
            name = name.split('.')
            while len(name):
                __o = getattr(__o, name.pop(0))
        else:
            __o = getattr(__o, name)
        return __o

    def set_attribute(self, name: str, obj: any):
        self._modules[name] = obj
        setattr(self, name, self._modules[name])

    def set_obj_module(self, name: str, module: any, set_attr: bool = True, **kwargs):
        """
        Set a module from a class and put it on self.modules, with kwargs
        :param set_attr: set as attribute of object or not
        :param name: module name
        :param module: class
        :param kwargs: all arguments
        :return:
        """
        if set_attr:
            self.set_attribute(name, module(self, **kwargs))
        else:
            self._modules[name] = module(self, **kwargs)

    def set_fn_module(self, name: str, fn_module, set_attr: bool = True, **kwargs):
        if set_attr:
            self.set_attribute(name, fn_module(**kwargs))
        else:
            self._modules[name] = fn_module(**kwargs)

    def select_item_or_all(self, name: any):
        """
        select a module name or list of selected name
        :param name: string or list
        :return:
        """
        if name is None:
            return self._modules.keys()
        else:
            assert isinstance(name, str) or isinstance(name, list), "Please give input of str or list!"
            if isinstance(name, str):
                assert name in self._modules.keys(), "Input name not valid !"
                return [name]
            elif isinstance(name, list):
                return [n for n in name if n in self._modules.keys()]

    def get(self, name, default=None):
        return self._modules.get(name, default)

    def __getitem__(self, name: str):
        return self._modules[name]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        for name in self._modules.keys():
            yield self._modules[name]

    # container properties ----------------------------------
    modules = property(get_modules, set_modules)
    configs = property(get_configs, set_configs)
    params = property(get_params, set_params)

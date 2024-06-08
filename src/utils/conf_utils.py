import utils.util_funcs as uf
from abc import abstractmethod, ABCMeta
import os
from utils.proj_settings import RES_PATH, SUM_PATH, TEMP_PATH

early_stop = 50
epochs = 200


class ModelConfig(metaclass=ABCMeta):
    """

    """

    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.seed = 0
        self.early_stop = early_stop
        self.epochs = epochs
        self.birth_time = uf.get_cur_time(t_format='%m_%d-%H_%M_%S')
        # Other attributes
        self._file_conf_list = ['checkpoint_file', 'res_file']
        self._ignored_settings = ['_ignored_settings', 'log_on', '_file_conf_list', 'device', 'n_class', 'n_feat', 'important_paras']

    def __str__(self):
        # Print all attributes including data and other path settings added to the config object.
        return str({k: v for k, v in self.model_conf.items()})

    def update_modified_conf(self, conf_dict):
        self.__dict__.update(conf_dict)
        uf.mkdir_list([getattr(self, _) for _ in self._file_conf_list])

    def mkdir(self, additional_files=[]):
        uf.mkdir_list([getattr(self, _) for _ in (self._file_conf_list + additional_files)])

    @property
    def f_prefix(self):
        return f"E{self.epochs}_{self.model_cf_str}"

    @property
    @abstractmethod
    def model_cf_str(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    def model_conf(self):
        # Print the model settings only.
        return {k: v for k, v in sorted(self.__dict__.items())
                if k not in self._ignored_settings}

    @property
    @abstractmethod
    def checkpoint_file(self):
        # Model config to str
        return ValueError('The checkpoint file name must be defined')

    @property
    def res_file(self):
        return f'{RES_PATH}{self.model}/{self.dataset}/{self.f_prefix}.txt'
        # return f'{RES_PATH}{self.model}/{self.dataset}/{self.f_prefix}.txt'

    def sub_conf(self, sub_conf_list):
        # Generate subconfig dict using sub_conf_list
        return {k: self.__dict__[k] for k in sub_conf_list}

    @staticmethod
    def add_exp_setting_args(parser):
        parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id to use, -1 for cpu")
        parser.add_argument("-d", "--dataset", type=str, default='IMDB-BINARY')
        parser.add_argument("-e", "--early_stop", default=early_stop, type=int)
        parser.add_argument('--log_on', action="store_false", help='show log or not')
        parser.add_argument("--epochs", default=epochs, type=int)
        parser.add_argument("--seed", default=1, type=int)
        return parser

    def __str__(self):
        return f'{self.model} config: \n{self.model_conf}'


class SimpleObject():
    """
    convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf={}):
        self.__dict__.update(conf)

import os
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Iterable

import colorama
import torch.cuda
import torch.distributed as pt_dist
from ruamel import yaml
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from pytorch_helper.utils.dist import get_rank
from pytorch_helper.utils.dist import is_distributed

colorama.init()

VERBOSE_INFO = 3  # all
VERBOSE_WARN = 2  # warn and error
VERBOSE_ERROR = 1  # error only
VERBOSE_NONE = 0  # verbose off

bar = tqdm
bar_len = 80
verbose_level = VERBOSE_INFO

if get_rank() != 0:
    verbose_level = VERBOSE_NONE

__all__ = [
    'notebook_compatible',
    'pbar',
    'info',
    'warn',
    'error',
    'pretty_dict',
    'get_datetime'
]


def notebook_compatible():
    """ setup the module to make it compatible with jupyter notebook
    """
    global bar
    bar = tqdm_notebook
    global bar_len
    bar_len = None
    global verbose_level
    verbose_level = VERBOSE_NONE


def pbar(iterable: Iterable = None, ncols: int = bar_len, **kwargs):
    """ create a tqdm bar

    :param iterable: iterable object
    :param ncols: int of progress bar length
    :param kwargs: extra keyword arguments to create the progress bar
    :return:
    """
    return bar(iterable, ncols=ncols, **kwargs)


def _get_prefix(level: int, field: str) -> str:
    """ generate the prefix for the logging

    :param level: int of verbose level
    :param field: str to indicate the logging position
    :return: str
    """
    prefix = colorama.Fore.GREEN + '['
    time_str = datetime.now().strftime('%m/%d %H:%M:%S')
    if level == VERBOSE_ERROR:
        prefix += f'{colorama.Fore.RED}ERROR{colorama.Fore.GREEN}'
    elif level == VERBOSE_WARN:
        prefix += f'{colorama.Fore.YELLOW}WARN{colorama.Fore.GREEN}]'
    else:
        prefix += f'INFO]'

    prefix += f'[{time_str}][{_get_device()}][{field}]: '
    prefix += colorama.Style.RESET_ALL
    return prefix


def info(field: str, msg: str, **kwargs: Dict[str, Any]):
    """ print an info message

    :param field: str to indicate the logging position
    :param msg: str of the logging content
    :param kwargs: extra keyword arguments posted to `write` function
    """
    if verbose_level >= VERBOSE_INFO:
        bar.write(_get_prefix(VERBOSE_INFO, field) + msg, **kwargs)


def warn(field: str, msg: str, **kwargs: Dict[str, Any]):
    """ print a warning message

    :param field: str to indicate the logging position
    :param msg: str of the logging content
    :param kwargs: extra keyword arguments posted to `write` function
    """
    if verbose_level >= VERBOSE_WARN:
        bar.write(_get_prefix(VERBOSE_WARN, field) + msg, **kwargs)


def error(field: str, msg: str, **kwargs: Dict[str, Any]):
    """ print an error message

    :param field: str to indicate the logging position
    :param msg: str of the logging content
    :param kwargs: extra keyword arguments posted to `write` function
    """
    if verbose_level >= VERBOSE_ERROR:
        bar.write(_get_prefix(VERBOSE_ERROR, field) + msg, **kwargs)


def pretty_dict(a: dict) -> str:
    """ convert dict `a` to str in pretty yaml format

    :param a: dict to convert
    :return: str
    """
    yaml_obj = yaml.YAML()
    yaml_obj.indent(mapping=4, sequence=4)

    class MySteam:
        def __init__(self):
            self.s = ''

        def write(self, s):
            self.s += s.decode('utf-8')

    stream = MySteam()
    yaml_obj.dump(a, stream)
    return stream.s


def _get_device() -> str:
    """ get the str of current GPU device
    """
    device = ''
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if is_distributed():
        rank = pt_dist.get_rank()
        gpu_id = visible_devices[rank]
        device = f'RANK{rank} on GPU{gpu_id}'
    elif torch.cuda.is_available():
        gpu_id = visible_devices[torch.cuda.current_device()]
        device = f'GPU{gpu_id}'
    return device


def get_datetime() -> str:
    """ get the str of current date and time
    """
    return datetime.now().strftime('%b%d_%H-%M-%S')

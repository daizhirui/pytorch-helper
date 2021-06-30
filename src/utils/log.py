import os
from datetime import datetime
from typing import Any
from typing import Dict

import torch.cuda
import torch.distributed as pt_dist
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

VERBOSE_INFO = 3  # all
VERBOSE_WARN = 2  # warn and error
VERBOSE_ERROR = 1  # error only
VERBOSE_NONE = 0  # verbose off

bar = tqdm
bar_len = 80
verbose_level = VERBOSE_INFO

if hasattr(pt_dist, 'is_initialized') and pt_dist.is_initialized():
    if pt_dist.get_rank() != 0:
        verbose_level = VERBOSE_NONE


def notebook_compatible():
    global bar
    bar = tqdm_notebook
    global bar_len
    bar_len = None
    global verbose_level
    verbose_level = VERBOSE_NONE


def info(msg: str, **kwargs: Dict[str, Any]):
    if verbose_level >= VERBOSE_INFO:
        bar.write(f'[INFO]{_get_device()}[{get_datetime()}]:{msg}', **kwargs)


def warn(msg: str, **kwargs: Dict[str, Any]):
    if verbose_level >= VERBOSE_WARN:
        bar.write(f'[WARN]{_get_device()}[{get_datetime()}]:{msg}', **kwargs)


def error(msg: str, **kwargs: Dict[str, Any]):
    if verbose_level >= VERBOSE_ERROR:
        bar.write(f'[ERROR]{_get_device()}[{get_datetime()}]:{msg}', **kwargs)


def pretty_dict(a: dict, prefix: str = '', msg: str = ''):
    max_len = max([len(k) for k in a])
    for k, v in a.items():
        if isinstance(v, dict):
            pretty_dict(v, prefix + ' ' * 4, msg)
        else:
            msg += f'{prefix}{k:>{max_len}} = {v}\n'
    return msg


def _get_device():
    device = ''
    if hasattr(pt_dist, 'is_initialized') and pt_dist.is_initialized():
        rank: int = pt_dist.get_rank()
        real_cuda_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank]
        device = f'[RANK{pt_dist.get_rank()} on CUDA{real_cuda_id}]'
    elif torch.cuda.is_available():
        device = f'[CUDA{torch.cuda.current_device()}]'
    return device


def get_datetime():
    return datetime.now().strftime('%b%d_%H-%M-%S')

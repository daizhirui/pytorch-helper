# Copyright (c) Zhirui Dai
import csv
import os.path
import pickle
import tarfile
import time
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import List
from typing import Union

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ruamel.yaml as yaml
import torch
from PIL import Image

from .log import info

img_ext = 'png'


def make_dirs(path: str):
    """ create directories of `path`

    :param path: str of the directory path to create
    """
    os.makedirs(os.path.abspath(path), exist_ok=True)


def make_dirs_for_file(path: str):
    """ create the folder for the file specified by `path`

    :param path: str of the file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_result(result: Any, path: str):
    """ save `result` as a pickle file to `path`

    :param result:
    :param path:
    :return:
    """
    make_dirs_for_file(path)
    with open(path, 'wb') as f:
        pickle.dump(result, f)


def load_result(path: str):
    """ load data from pickle file

    :param path: str of the path of the pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_yaml(path: str) -> dict:
    """ load dict from yaml file

    :param path: str of the path of the yaml file
    :return: dict of the yaml file
    """
    with open(path, 'r') as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict


def load_txt(path: str) -> List[str]:
    """ load lines from text file

    :param path: str of the path of the text file
    :return: list of lines in the file
    """
    with open(path, 'r') as f:
        return [x.strip() for x in f.readlines()]


def load_pth(path: str) -> dict:
    """ load state dict from a checkpoint file

    :param path: str of path of the checkpoint file
    :return: dict
    """
    info(__name__, f"Load state dict from {path}")
    t = time.time()
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    t = time.time() - t
    info(__name__, f'Loaded after {t:.2f} seconds')
    return state_dict


def save_pth(path: str, state_dict: dict):
    """ save `state_dict` as a checkpoint file to `path`

    :param path: str of the path to save the checkpoint
    :param state_dict: dict
    """
    make_dirs_for_file(path)
    if torch.__version__ >= '1.6.0':
        torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    else:
        torch.save(state_dict, path)


def imsave(path: str, arr: Union[Image.Image, plt.Figure, np.ndarray]):
    """ save image `arr` to `path`

    :param path: str of the path to save the image
    :param arr: array of the image
    """
    path = f'{path}.{img_ext}'
    make_dirs_for_file(path)
    if isinstance(arr, Image.Image):
        arr.save(path)
    elif isinstance(arr, plt.Figure):
        arr.savefig(path)
    else:
        if np.max(arr) <= 1:
            arr = (arr * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    info(__name__, f'Save to {path}')


def save_dict_as_csv(a: OrderedDict, path: str, append=False):
    """ save dict `a` as a csv file to `path`

    :param a: OrderedDict to save
    :param path: str of the csv file path
    :param append: Bool to append `a` as a row to the file at `path`
    """
    make_dirs_for_file(path)

    new_csv = True
    fieldnames = list(a.keys())
    if os.path.exists(path) and append:
        new_csv = False
        with open(path, 'r') as file:
            old_fieldnames = file.readline().strip().split(',')
            diff = set(fieldnames).difference(old_fieldnames)
            if len(diff) > 0:
                new_csv = True
                fieldnames += sorted(list(diff))
    else:
        fieldnames = list(a.keys())

    with open(path, 'a+' if append else 'w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        if new_csv:
            csv_writer.writeheader()
        for k in fieldnames:
            if k not in a:
                a[k] = 'None'
        csv_writer.writerow(a)


def make_tar_file(src: str, dst: str, include: Callable = None):
    """ make a tar file from `src` to `dst`

    :param src: str of the path of source
    :param dst: str of the path to save the tar file
    :param include: Callable to determine which files to include in the tar file
    """
    with tarfile.open(dst, 'w:gz') as file:
        file.add(src, arcname=os.path.basename(src), filter=include)

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .log import info

img_ext = 'png'


def save_result(result, path):
    with open(path, 'wb') as f:
        pickle.dump(result, f)


def load_result(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_txt(path):
    with open(path, 'r') as f:
        return [x.strip() for x in f.readlines()]


def load_pth(path):
    info(f"Load state dict from {path}")
    t = time.time()
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    t = time.time() - t
    info(f'Loaded after {t:.2f} seconds')
    return state_dict


def save_pth(path, state_dict):
    if torch.__version__ >= '1.6.0':
        torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    else:
        torch.save(state_dict, path)


def imsave(path, arr):
    path = f'{path}.{img_ext}'
    if isinstance(arr, Image.Image):
        arr.save(path)
    elif isinstance(arr, plt.Figure):
        arr.savefig(path)
    else:
        if arr.max() <= 1:
            arr = (arr * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    info(f'Save to {path}')

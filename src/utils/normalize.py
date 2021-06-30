from typing import Optional
from typing import Union

import numpy as np
from numpy import ndarray
from torch import Tensor


def normalize(arr: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
    """ normalize elements in arr to [0, 1]

    :param arr: the matrix to normalize
    :return: the same type as arr
    """
    amin = arr.min()
    diff = arr.max() - amin + 1e-8
    return (arr - amin) / diff


def normalize_sum(arr: ndarray, axis: Optional[int] = None) -> ndarray:
    # axis=None: by all
    # axis=0: by column
    # axis=1: by row
    assert np.nan not in arr
    s = np.sum(arr, axis=axis, keepdims=True)
    return arr.astype(np.float32) / s

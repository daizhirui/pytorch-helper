from typing import Union

import numpy as np
from numpy import ndarray
from torch import Tensor


def to_numpy(arr: Union[ndarray, Tensor], np_type=np.float32) -> ndarray:
    """ convert `arr` to numpy array

    :param arr: `numpy.ndarray` or `torch.Tensor`
    :param np_type: data type
    :return: numpy.ndarray
    """
    if isinstance(arr, ndarray):
        return arr.astype(np_type)
    elif isinstance(arr, Tensor):
        return arr.data.cpu().numpy().astype(np_type)
    else:
        raise ValueError(f"Unknown data type: {np_type(arr)}")

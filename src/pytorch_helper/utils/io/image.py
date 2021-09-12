import io
from typing import Any
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from . import config
from .make_dirs import make_dirs_for_file
from ..log import get_logger

__all__ = [
    'imread',
    'imsave',
    'plt_figure_to_numpy',
    'plt_figure_to_pil',
    'image_obj_to_numpy'
]

logger = get_logger(__name__)


def imread(path: str) -> np.ndarray:
    """ read image from file `path`

    :param path: str of the file path
    :return: numpy array of the image
    """
    return plt.imread(path)


def imsave(
    path: str, arr: Union[Image.Image, plt.Figure, np.ndarray],
    img_ext: str = None
):
    """ save image `arr` to `path`

    :param path: str of the path to save the image
    :param arr: array of the image
    :param img_ext: extension of the image file, default png
    """
    if img_ext is None:
        img_ext = config['img_ext']
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
    logger.info(f'Save {path}')


def plt_figure_to_numpy(figure: plt.Figure = None) -> np.ndarray:
    """ convert a `matplotlib.pyplot.Figure` to numpy array

    :param figure: matplotlib.pyplot.Figure to convert
    :return: numpy array of the figure as a RGBA image, whose dtype is uint8
    """
    if figure is None:
        figure = plt.gcf()
    agg = figure.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()

    arr = np.array(figure.canvas.renderer.buffer_rgba())
    return arr


def plt_figure_to_pil(figure: plt.Figure = None) -> Image:
    """ convert a `matplotlib.pyplot.Figure` to `PIL.Image.Image`

    :param figure: matplotlib.pyplot.Figure to convert
    :return: PIL.Image.Image of the plot
    """
    if figure is None:
        figure = plt.gcf()
    agg = figure.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()

    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def image_obj_to_numpy(img_obj: Any) -> np.ndarray:
    """ convert an image object to a numpy array

    :param img_obj: image object to convert
    :return: numpy array of the image
    """
    if isinstance(img_obj, plt.Figure):
        return plt_figure_to_numpy(img_obj)
    else:
        return np.array(img_obj)

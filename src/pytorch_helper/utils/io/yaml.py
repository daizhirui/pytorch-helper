from ruamel import yaml

from .make_dirs import make_dirs_for_file
from ..log import info

__all__ = [
    'load_yaml',
    'save_yaml'
]


def load_yaml(path: str) -> dict:
    """ load dict from yaml file

    :param path: str of the path of the yaml file
    :return: dict of the yaml file
    """
    info(__name__, f'Load from {path}')
    with open(path, 'r') as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict


def save_yaml(path: str, a: dict):
    """ save dict as a yaml file

    :param path: str of the file path
    :param a: dict to save
    """
    make_dirs_for_file(path)

    yaml_obj = yaml.YAML()
    yaml_obj.indent(mapping=4, sequence=4)
    with open(path, 'w') as file:
        yaml_obj.dump(a, file)
    info(__name__, f'Save {path}')

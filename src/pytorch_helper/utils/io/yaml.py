from ruamel import yaml
import os
from .make_dirs import make_dirs_for_file
from ..log import get_logger

__all__ = [
    'load_yaml',
    'save_yaml'
]

logger = get_logger(__name__)


def load_yaml(
    path: str,
    recursive: bool = True,
    recursive_mark: str = '<<'
) -> dict:
    """ load dict from yaml file

    :param path: str of the path of the yaml file
    :param recursive:
    :param recursive_mark:
    :return: dict of the yaml file
    """
    logger.info(f'Load from {path}')
    with open(path, 'r') as file:
        a = yaml.safe_load(file)

    if not recursive:
        return a

    def _check_dict(d: dict):
        for k, v in d.items():
            if isinstance(v, str):
                if v.startswith(recursive_mark):
                    v = v[2:].strip()
                    v_path = os.path.join(os.path.dirname(path), v)
                    logger.info(f'Load key {k} from {v} recursively')
                    d[k] = load_yaml(v_path, recursive, recursive_mark)
            elif isinstance(v, dict):
                d[k] = _check_dict(v)
            elif isinstance(v, list):
                d[k] = _check_list(v)
        return d

    def _check_list(d):
        for i in range(len(d)):
            v = d[i]
            if isinstance(v, str):
                if v.startswith(recursive_mark):
                    v = v[2:].strip()
                    v_path = os.path.join(os.path.dirname(path), v)
                    logger.info(f'Load list item{i} {v} recursively')
                    d[i] = load_yaml(v_path, recursive, recursive_mark)
            elif isinstance(v, dict):
                d[i] = _check_dict(v)
            elif isinstance(v, list):
                d[i] = _check_list(v)
        return d

    check_func = {
        _check_dict.__name__: _check_dict,
        _check_list.__name__: _check_list
    }

    a = check_func.get(f'_check_{type(a).__name__}', lambda x: x)(a)

    return a


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
    logger.info(f'Save {path}')

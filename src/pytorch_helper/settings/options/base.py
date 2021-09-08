import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from typing import Type
from typing import TypeVar

import ruamel.yaml as yaml

from ...utils.io import save_yaml
from ...utils.log import get_logger

T = TypeVar('T')

__all__ = ['OptionBase']

logger = get_logger(__name__)


class _Base:
    option_file_dir: str = None

    # implement class methods
    @classmethod
    def load_from_dict(cls: Type[T], option_dict: dict, **kwargs) -> T:
        """ Build option from a dict of arguments

        :param option_dict: Dict of arguments
        :param kwargs: extra keyword arguments
        :return: T instance
        """
        if option_dict is None:
            return None
        option_dict.update(kwargs)
        logger.info(f'create {cls.__name__} from dict')
        return cls(**option_dict)

    @classmethod
    def load_from_file(cls: Type[T], option_file: str, **kwargs) -> T:
        """ Build option from a yaml file

        :param option_file: Path to the yaml file
        :param kwargs: extra keyword arguments
        :return: T instance
        """
        if option_file is None:
            return None
        with open(option_file, 'r') as file:
            option_dict: dict = yaml.safe_load(file)
        option_dict.update(kwargs)
        logger.info(f'create {cls.__name__} from file {option_file}')
        return cls(**option_dict)


@dataclass()
class OptionBase(_Base):

    def __post_init__(self):
        if OptionBase.option_file_dir is None:
            logger.warn('OptionBase.option_file_dir is not set, skip loading '
                        'from file recursively.')
            return
        for f in fields(self):
            attr = getattr(self, f.name)
            if f.type is type(attr):
                continue
            if not isinstance(attr, str):
                continue
            path = os.path.join(OptionBase.option_file_dir, attr)
            if os.path.exists(path):
                with open(path, 'r') as file:
                    setattr(self, f.name, yaml.safe_load(file))
                logger.info(f'Load {type(self).__name__}.{f.name} from {path}')

    def asdict(self) -> dict:
        """ convert self to Dict

        :return: Dict of self's attributes
        """
        return asdict(self)

    def save_as_yaml(self, output_path: str):
        """ save self as a yaml file

        :param output_path: path to output
        :return:
        """
        save_yaml(output_path, self.asdict())

    def __str__(self):
        """ convert self to a pretty yaml string the same as the content in the
        file created by `self.save_as_yaml`

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
        yaml_obj.dump(self.asdict(), stream)
        return stream.s

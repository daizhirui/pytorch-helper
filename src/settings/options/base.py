from dataclasses import asdict
from dataclasses import dataclass
from typing import Type
from typing import TypeVar

import ruamel.yaml as yaml

from utils.log import pretty_dict

T = TypeVar('T')


class _Base:
    @classmethod
    def load_from_dict(cls: Type[T], option_dict: dict, **kwargs) -> T:
        if option_dict is None:
            return None
        print(f"Options of {cls.__name__}:\n{pretty_dict(option_dict)}")
        option_dict.update(kwargs)
        return cls(**option_dict)

    @classmethod
    def load_from_file(cls: T, option_file: str, **kwargs) -> T:
        if option_file is None:
            return None
        with open(option_file, 'r') as file:
            option_dict: dict = yaml.safe_load(file)
        print(f"Options of {cls.__name__}:\n{pretty_dict(option_dict)}")
        option_dict.update(kwargs)
        return cls(**option_dict)


@dataclass()
class OptionBase(_Base):
    def save(self, output_path: str):
        yaml_obj = yaml.YAML()
        yaml_obj.indent(mapping=4, sequence=4)
        option_dict = asdict(self)
        with open(output_path, 'w') as file:
            yaml_obj.dump(option_dict, file)

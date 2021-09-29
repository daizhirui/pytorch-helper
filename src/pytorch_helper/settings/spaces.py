from collections import defaultdict
from enum import Enum


class Spaces:
    _registry = defaultdict(dict)

    class NAME(Enum):
        MODEL = 'model'
        DATALOADER = 'dataloader'
        LOSS_FN = 'loss_fn'
        METRIC = 'metric'
        TASK_OPTION = 'task_option'
        TASK_FOR_TRAIN = 'task_for_train'
        TASK_FOR_TEST = 'task_for_test'

    def __init_subclass__(cls, **kwargs):
        space = kwargs.pop('space')
        space = Spaces.NAME(space)
        ref = kwargs.pop('ref')
        super().__init_subclass__(**kwargs)

        cls._registry[space][ref] = cls

    @classmethod
    def add_space(cls, name, value):
        a = {x.name: x.value for x in cls.NAME}
        a[name] = value
        cls.NAME = Enum('NAME', a)

    @staticmethod
    def build(space, ref, *args, **kwargs):
        space = Spaces.NAME(space)
        return Spaces._registry[space][ref](*args, **kwargs)

    @staticmethod
    def register(cls, spaces, refs):
        for space, ref in zip(spaces, refs):
            space = Spaces.NAME(space)
            Spaces._registry[space][ref] = cls

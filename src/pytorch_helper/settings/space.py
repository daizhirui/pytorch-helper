from typing import Any

__all__ = ['Spaces']


class Spaces:
    """ Spaces is a class used to store the mapping from a str to a class.
    Some classes can be used for different tasks, different models, etc. Spaces
    is used to simplify the coding. Besides, Spaces help to make sure that a
    correct class is used for a specified name.
    """

    MODEL = 'models'
    DATALOADER = 'dataloaders'
    LOSS_FN = 'loss_fns'
    METRICS = 'metrics'
    TASK_OPTIONS = 'task_options'
    TASK_FOR_TRAIN = 'tasks_for_train'
    TASK_FOR_TEST = 'tasks_for_test'

    _spaces = {
        MODEL: dict(),
        DATALOADER: dict(),
        LOSS_FN: dict(),
        METRICS: dict(),
        TASK_OPTIONS: dict(),
        TASK_FOR_TRAIN: dict(),
        TASK_FOR_TEST: dict()
    }

    @staticmethod
    def new_space(name: str, attr: str = None):
        """ create a new space to store mappings

        :param name: the space name
        :param attr: name of the attribute to store `name`
        :return:
        """
        if name in Spaces._spaces.keys():
            raise ValueError(f'space name: "{name}" already exists')

        if attr is None:
            attr = name
        Spaces._spaces[name] = dict()
        setattr(Spaces, attr, name)
        return Spaces._spaces[name]

    @staticmethod
    def get_space(space_name: str) -> dict:
        """ get the space with the specified space name

        :param space_name: the name of space
        :return: dict
        """
        return Spaces._spaces[space_name]

    @staticmethod
    def register(cls, space_name: str, cls_ref: str = None):
        """ register `cls` to `cls_ref` in space of name `space_name`

        :param cls: the class to register
        :param space_name: the name of the space
        :param cls_ref: str of reference to `cls`
        :return:
        """
        if cls_ref is None:
            cls_ref = cls.__name__
        Spaces.get_space(space_name)[cls_ref] = cls

    @staticmethod
    def register_model(model_cls):
        """ register `model_cls` in the model space

        :param model_cls: class of the model
        """
        Spaces.register(model_cls, Spaces.MODEL)

    @staticmethod
    def register_dataloader(dataloader_cls):
        """ register `dataloader_cls` in the dataloader space

        :param dataloader_cls: class of the dataloader
        """
        Spaces.register(dataloader_cls, Spaces.DATALOADER)

    @staticmethod
    def register_loss_fn(loss_fn_cls):
        """ register `loss_fn` in the loss function space

        :param loss_fn_cls: class of the loss function
        """
        Spaces.register(loss_fn_cls, Spaces.LOSS_FN)

    @staticmethod
    def register_metric(metric_cls):
        """ register `metric_cls` in the metric space

        :param metric_cls: class of the metric
        """
        Spaces.register(metric_cls, Spaces.METRICS)

    @staticmethod
    def register_task_option(task_option_cls, ref: str):
        """ register `task_option_cls` to `ref` in the task option space

        :param task_option_cls: class of the task option
        :param ref: str as a reference
        """
        Spaces.register(task_option_cls, Spaces.TASK_OPTIONS, ref)

    @staticmethod
    def register_task_for_train(task_cls, ref: str):
        """ register `task_cls` to `ref` in the training task space

        :param task_cls: class of task
        :param ref: str as a reference
        """
        Spaces.register(task_cls, Spaces.TASK_FOR_TRAIN, ref)

    @staticmethod
    def register_task_for_test(task_cls, ref: str):
        """ register `task_cls` to `ref` in the test task space

        :param task_cls: class of task
        :param ref: str as a reference
        """
        Spaces.register(task_cls, Spaces.TASK_FOR_TEST, ref)

    @staticmethod
    def build(space_name: str, ref: str, *args, **kwargs) -> Any:
        """ build an instance of class referred as `name` in the space of
        `space_name`

        :param space_name: the name of the space to find the class
        :param ref: str to refer the class
        :param args: position arguments used to build the class instance
        :param kwargs: keyword arguments used to build the class instance
        """
        return Spaces._spaces[space_name][ref](*args, **kwargs)

    @staticmethod
    def build_model(model_name: str, *args, **kwargs):
        """ build a model referred as `model_name`

        :param model_name: reference of the model class
        :param args: position arguments used to build the model
        :param kwargs: keyword arguments used to build the model
        :return: model instance
        """
        return Spaces.build(Spaces.MODEL, model_name, *args, **kwargs)

    @staticmethod
    def build_dataloader(loader_name: str, *args, **kwargs):
        """ build a dataloader referred as `loader_name`

        :param loader_name: reference of the dataloader class
        :param args: position arguments used to build the dataloader
        :param kwargs: keyword arguments used to build the dataloader
        :return: dataloader instance
        """
        return Spaces.build(Spaces.DATALOADER, loader_name, *args, **kwargs)

    @staticmethod
    def build_loss_fn(loss_name: str, *args, **kwargs):
        """ build a loss function referred as `loss_name`

        :param loss_name: reference of the loss function
        :param args: position arguments used to build the loss function
        :param kwargs: keyword arguments used to build the loss function
        :return: loss function instance
        """
        return Spaces.build(Spaces.LOSS_FN, loss_name, *args, **kwargs)

    @staticmethod
    def build_metric(metric_name: str, *args, **kwargs):
        """ build a metric referred as `metric_name`

        :param metric_name: reference of metric
        :param args: position arguments used to build the metric
        :param kwargs: keyword arguments used to build the metric
        :return: metric instance
        """
        return Spaces.build(Spaces.METRICS, metric_name, *args, **kwargs)

    @staticmethod
    def build_task_option(task_dict: dict):
        """ build a task option from `task_dict`

        :param task_dict: dict used to build task option
        :return: a task option instance
        """
        task_option_cls = Spaces._spaces[Spaces.TASK_OPTIONS][task_dict['type']]
        return task_option_cls.load_from_dict(task_dict)

    @staticmethod
    def build_task(option):
        """ build a task from `option`

        :param option: task option instance
        :return: a task instance
        """
        task_type = option.type
        if option.train:
            space_name = Spaces.TASK_FOR_TRAIN
        else:
            space_name = Spaces.TASK_FOR_TEST
        task_cls = Spaces._spaces[space_name][task_type]
        return task_cls(option)

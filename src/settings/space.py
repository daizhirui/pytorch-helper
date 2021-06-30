class Spaces:
    MODEL = 'models'
    DATALOADER = 'dataloaders'
    LOSS_FN = 'loss_fns'
    METRICS = 'metrics'
    TASK_OPTIONS = 'task_options',
    TASK_FOR_TRAIN = 'tasks_for_train'
    TASK_FOR_TEST = 'tasks_for_test'

    _spaces = {
        MODEL         : dict(),
        DATALOADER    : dict(),
        LOSS_FN       : dict(),
        METRICS       : dict(),
        TASK_OPTIONS  : dict(),
        TASK_FOR_TRAIN: dict(),
        TASK_FOR_TEST : dict()
    }

    @staticmethod
    def new_space(name, key=None):
        Spaces._spaces[name] = dict()
        if key is None:
            key = name
        setattr(Spaces, name, key)
        return Spaces._spaces[name]

    @staticmethod
    def get_space(space_name):
        return Spaces._spaces[space_name]

    @staticmethod
    def register(cls, space_name, cls_ref=None):
        if cls_ref is None:
            cls_ref = cls.__name__
        Spaces.get_space(space_name)[cls_ref] = cls

    @staticmethod
    def register_model(model_cls):
        Spaces.register(model_cls, Spaces.MODEL)

    @staticmethod
    def register_dataloader(dataloader_cls):
        Spaces.register(dataloader_cls, Spaces.DATALOADER)

    @staticmethod
    def register_loss_fn(loss_fn_cls):
        Spaces.register(loss_fn_cls, Spaces.LOSS_FN)

    @staticmethod
    def register_metric(metric_cls):
        Spaces.register(metric_cls, Spaces.METRICS)

    @staticmethod
    def register_task_option(task_option_cls, ref):
        Spaces.register(task_option_cls, Spaces.TASK_OPTIONS, ref)

    @staticmethod
    def register_task_for_train(task_cls, ref):
        Spaces.register(task_cls, Spaces.TASK_FOR_TRAIN, ref)

    @staticmethod
    def register_task_for_test(task_cls, ref):
        Spaces.register(task_cls, Spaces.TASK_FOR_TEST, ref)

    @staticmethod
    def build(space_name, name, *args, **kwargs):
        return Spaces._spaces[space_name][name](*args, **kwargs)

    @staticmethod
    def build_model(model_name, *args, **kwargs):
        return Spaces.build(Spaces.MODEL, model_name, *args, **kwargs)

    @staticmethod
    def build_dataloader(loader_name, *args, **kwargs):
        return Spaces.build(Spaces.DATALOADER, loader_name, *args, **kwargs)

    @staticmethod
    def build_loss_fn(loss_name, *args, **kwargs):
        return Spaces.build(Spaces.LOSS_FN, loss_name, *args, **kwargs)

    @staticmethod
    def build_metric(metric_name, *args, **kwargs):
        return Spaces.build(Spaces.METRICS, metric_name, *args, **kwargs)

    @staticmethod
    def build_task_option(task_dict):
        task_option_cls = Spaces._spaces[Spaces.TASK_OPTIONS][task_dict['type']]
        return task_option_cls.load_from_dict(task_dict)

    @staticmethod
    def build_task(option, gpu_ids):
        task_type = option.type
        if option.train:
            space_name = Spaces.TASK_FOR_TRAIN
        else:
            space_name = Spaces.TASK_FOR_TEST
        task_cls = Spaces._spaces[space_name][task_type]
        return task_cls(option, gpu_ids)

import importlib
import yaml


def make_instance(cls_path, *args, **kwargs):
    """Функция реализует инициализацию инстанса произвольного класса по конфигу.

    Args:
        cls_path: путь до класса из репозитория neural_search
    """
    cls_path = cls_path.split(".")
    module_path = ".".join(cls_path[:-1])
    cls_name = cls_path[-1]
    module = importlib.import_module(f"lib.{module_path}")
    cls_object = getattr(module, cls_name)
    return cls_object(*args, **kwargs)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def seed_everything(seed=0):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

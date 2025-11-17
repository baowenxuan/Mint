from .vision import VISION_DATASETS, get_vision_dataset_class
from .corruption import CORRUPTION_DATASETS, get_corruption_dataset_class
from .imagenets import IMAGENET_VARIANT_DATASETS, get_imagenet_variant_dataset_class


def get_dataset_class(dataset_name):
    if dataset_name in VISION_DATASETS:
        return get_vision_dataset_class(dataset_name)

    elif dataset_name in CORRUPTION_DATASETS:
        return get_corruption_dataset_class(dataset_name)

    elif dataset_name in IMAGENET_VARIANT_DATASETS:
        return get_imagenet_variant_dataset_class(dataset_name)

    else:
        raise Exception("Unknown dataset name")

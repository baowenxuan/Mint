from .ImageNetA import ImageNetA
from .ImageNetR import ImageNetR
from .ImageNetSketch import ImageNetSketch
from .ImageNetV2 import ImageNetV2


IMAGENET_VARIANT_DATASETS = [
    "ImageNetA",
    "ImageNetR",
    "ImageNetV2",
    "ImageNetSketch",
]


def get_imagenet_variant_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]
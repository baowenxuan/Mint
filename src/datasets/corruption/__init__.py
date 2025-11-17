from .CIFAR10C import CIFAR10C
from .CIFAR100C import CIFAR100C
from .ImageNetC import ImageNetC
from .ImageNetC5K import ImageNetC5K

CORRUPTION_DATASETS = [
    # 224 x 224 images
    "ImageNetC",
    "ImageNetC5K",  # robustbench subsampling

    # 32 x 32 images
    "CIFAR10C",
    "CIFAR100C",
]


def get_corruption_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

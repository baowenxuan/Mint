import os
import torchvision.datasets
import torchvision.transforms
import torch.nn as nn

from collections import OrderedDict

from .utils import CustomImageFolder

VISION_DATASETS = [
    "CIFAR10",
    "CIFAR100",
    "TinyImageNet",
    "ImageNet",
    "ImageNet5K",
]


class SingleDataset:
    def __init__(self):
        self.datasets = []

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class CIFAR10(SingleDataset):
    def __init__(self, root, transform):
        root = os.path.join(root, "torchvision")
        self.datasets = [torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)]

        self.environments = ['test']

        self.classes = self.datasets[0].classes

        self.num_classes = len(self.classes)


class CIFAR100(SingleDataset):
    def __init__(self, root, transform):
        root = os.path.join(root, "torchvision")
        self.datasets = [torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)]

        self.environments = ['test']

        self.classes = self.datasets[0].classes

        self.num_classes = len(self.classes)


def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames


class TinyImageNet(SingleDataset):
    """
    Clean version of TinyImageNetC
    According to the intruction here: https://github.com/hendrycks/robustness/issues/39
    Downloaded from https://drive.google.com/file/d/1pTkzhONG8o2Zx9ocYqjmqY3G5ejtzFjr/view

    """
    def __init__(self, root, transform):
        root = os.path.join(root, "torchvision", "TinyImageNet")
        path = os.path.join(root, "imagenet_val_bbox_crop")

        if transform:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(64),
                transform,
            ])
        else:
            transform = torchvision.transforms.Resize(64)

        self.datasets = [torchvision.datasets.ImageFolder(root=path, transform=transform)]

        self.environments = ['val']

        classnames = read_classnames(os.path.join(root, 'classnames.txt'))

        self.num_classes = len(self.datasets[-1].classes)
        self.classes = [classnames[cls] for cls in self.datasets[-1].classes]  # class names


class ImageNet(SingleDataset):
    def __init__(self, root, transform):
        root = os.path.join(root, "torchvision", "ImageNet")
        path = os.path.join(root, "val")

        if transform:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                transform
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224)
            ])

        self.datasets = [torchvision.datasets.ImageFolder(root=path, transform=transform)]

        self.environments = ['val']

        classnames = read_classnames(os.path.join(root, 'classnames.txt'))

        self.num_classes = len(self.datasets[-1].classes)
        self.classes = [classnames[cls] for cls in self.datasets[-1].classes]  # class names


class ImageNet5K(SingleDataset):
    def __init__(self, root, transform):
        root = os.path.join(root, "torchvision", "ImageNet")
        path = os.path.join(root, "val")

        if transform:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                transform
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224)
            ])

        self.datasets = [CustomImageFolder(root=path, transform=transform)]

        self.environments = ['val']

        classnames = read_classnames(os.path.join(root, 'classnames.txt'))

        self.num_classes = len(self.datasets[-1].classes)
        self.classes = [classnames[cls] for cls in self.datasets[-1].classes]  # class names


def get_vision_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

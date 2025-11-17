import os
import torch
from PIL import Image, ImageFile
from torchvision.datasets import ImageFolder
import numpy as np

from collections import OrderedDict

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Use the order in most papers
main_corruptions = [
    # Noise
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',

    # Blur
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',

    # Weather
    'snow',
    'frost',
    'fog',
    'brightness',

    # Digital
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
]

extra_corruptions = [
    'speckle_noise',
    'gaussian_blur',
    'spatter',
    'saturate',
]

all_corruptions = main_corruptions + extra_corruptions


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


class MultipleCorruptionImageFolder:

    def __init__(self, root, extra=False, severity=5, transform=None):

        if extra:
            self.environments = all_corruptions
        else:
            self.environments = main_corruptions

        self.datasets = []

        for i, environment in enumerate(self.environments):
            path = os.path.join(root, environment, str(severity))
            env_dataset = ImageFolder(path, transform=transform)
            self.datasets.append(env_dataset)

        classnames = read_classnames(os.path.join(root, 'classnames.txt'))

        self.num_classes = len(self.datasets[-1].classes)
        self.classes = [classnames[cls] for cls in self.datasets[-1].classes]  # class names

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class NumpyImageDataset:

    def __init__(self, X, Y, transform=None):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        if transform:
            self.transform = transform

        else:
            self.transform = lambda x: x  # identity mapping

    def __getitem__(self, item):
        return self.transform(Image.fromarray(self.X[item], mode='RGB')), int(self.Y[item])

    def __len__(self):
        return len(self.X)


class MultipleCorruptionNumpyImageDataset:
    def __init__(self, root, extra=False, severity=5, transform=None):

        if extra:
            self.environments = all_corruptions
        else:
            self.environments = main_corruptions

        self.datasets = []

        label_path = os.path.join(root, f"labels.npy")
        label = np.load(label_path)

        num_data = len(label) // 5  # since there are severity 1 - 5
        label = label[num_data * (severity - 1): num_data * severity]

        for i, environment in enumerate(self.environments):
            feat_path = os.path.join(root, f"{environment}.npy")
            feat = np.load(feat_path)

            feat = feat[num_data * (severity - 1): num_data * severity]  # use the corresponding severity

            env_dataset = NumpyImageDataset(feat, label, transform)
            self.datasets.append(env_dataset)

        self.set_classes()

    def set_classes(self):
        # implemented in Subclass
        raise NotImplementedError

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
import os

from .utils import all_corruptions, main_corruptions, read_classnames
from ..utils import CustomImageFolder


class ImageNetC5K:

    def __init__(self, root, extra=False, severity=5, transform=None):
        root = os.path.join(root, "corruption", "ImageNet-C")

        if extra:
            self.environments = all_corruptions
        else:
            self.environments = main_corruptions

        self.datasets = []

        for i, environment in enumerate(self.environments):
            path = os.path.join(root, environment, str(severity))
            env_dataset = CustomImageFolder(path, transform=transform)
            self.datasets.append(env_dataset)

        classnames = read_classnames(os.path.join(root, 'classnames.txt'))

        self.num_classes = len(self.datasets[-1].classes)
        self.classes = [classnames[cls] for cls in self.datasets[-1].classes]  # class names

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
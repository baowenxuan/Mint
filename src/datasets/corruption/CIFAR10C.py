import os

from .utils import MultipleCorruptionNumpyImageDataset


class CIFAR10C(MultipleCorruptionNumpyImageDataset):

    def __init__(self, root, extra=False, severity=5, transform=None):
        root = os.path.join(root, "corruption", "CIFAR-10-C")
        MultipleCorruptionNumpyImageDataset.__init__(self, root, extra, severity, transform)

    def set_classes(self):
        self.classes = ['airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']

        self.num_classes = len(self.classes)

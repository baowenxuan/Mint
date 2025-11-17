import os

from .utils import MultipleCorruptionImageFolder


class ImageNetC(MultipleCorruptionImageFolder):

    def __init__(self, root, extra=False, severity=5, transform=None):
        root = os.path.join(root, "corruption", "ImageNet-C")
        MultipleCorruptionImageFolder.__init__(self, root, extra, severity, transform)
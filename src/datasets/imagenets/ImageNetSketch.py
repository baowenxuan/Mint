import os
import torchvision


from .utils import SingleDataset, read_classnames


class ImageNetSketch(SingleDataset):
    """
    https://github.com/HaohanWang/ImageNet-Sketch
    """
    def __init__(self, root, transform):
        root = os.path.join(root, "imagenets", "imagenet-sketch")
        path = os.path.join(root, "sketch")

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
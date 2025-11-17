import os
import torchvision


from .utils import SingleDataset, read_classnames


class ImageNetR(SingleDataset):
    """
    https://github.com/hendrycks/imagenet-r
    """
    def __init__(self, root, transform):
        root = os.path.join(root, "imagenets", "imagenet-rendition")
        path = os.path.join(root, "imagenet-r")

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
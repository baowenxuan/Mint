from .CLIP import CLIP
from .Ensemble import Ensemble
from .Mint import Mint


def get_method_class(method_name):
    if method_name not in globals():
        raise NotImplementedError("Method not found: {}".format(method_name))
    return globals()[method_name]
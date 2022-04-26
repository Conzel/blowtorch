from ._convolutions import Conv2d, Conv2dTranspose
from ._relu import Relu
from ._interfaces import Layer

LAYER_DISPATCH = {"Conv2d": Conv2d,
                  "Conv2dTranspose": Conv2dTranspose, "ReLU": Relu}


def parse_layer(layer_spec: dict) -> Layer:
    return LAYER_DISPATCH[layer_spec["type"]](layer_spec)


class Model():
    """
    TODO
    Model that can be parsed by the models_template.rs Jinja file.
    """

    def __init__(self, specification):
        """
        TODO
        """
        self.module_name = specification["module_name"]
        self.layers = list(map(parse_layer, specification["layers"]))

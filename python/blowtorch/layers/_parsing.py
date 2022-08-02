from typing import List
from ._convolutions import Conv2d, Conv2dTranspose
from ._relu import Relu
from ._linear import LinearLayer
from ._flatten import Flatten
from ._interfaces import Layer

"""Contains the mapping of layer names (as in the specification) to 
the python classes that implement them."""
LAYER_DISPATCH = {
    "Conv2d": Conv2d,
    "Conv2dTranspose": Conv2dTranspose,
    "Linear": LinearLayer,
    "Flatten": Flatten,
    "ReLU": Relu,
}


def parse_layer(layer_spec: dict) -> Layer:
    """Parses the type of the layer from the specification and returns the corresponding
    python implementation of the layer that can be rendered."""
    return LAYER_DISPATCH[layer_spec["type"]](layer_spec)


class Model:
    """
    Model that can be parsed by the models_template.rs Jinja file.
    Models that we represent are always equivalent to simple Sequential Modules.

    Attributes:
        input_dim: The dimension of the input to the model (e.g. 3 for RGB images).
        output_dim: The dimension of the output of the model (e.g. 1 for a classification task)
        module_name: The name of the module that will be generated (this directly the name of the Rust/Python classes)
        layers: List of the layers that make up this model.
    """

    def __init__(self, specification):
        """
        Initializes a model that can be rendered by the models_template.rs Jinja file.
        """
        # for now we fix the input and output dimension, but this should be configurable
        # (think of a layer that flattens the input?)
        self.input_dim = 3
        self.output_dim = 1
        self.module_name = specification["module_name"]
        self.layers: List[Layer] = list(map(parse_layer, specification["layers"]))

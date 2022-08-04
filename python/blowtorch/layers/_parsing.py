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

"""
Contains the input and output shapes for the layers. -1 is a wildcard (takes 
output / input dimension of last / next layer)
"""
LAYER_INPUT_OUTPUT_SHAPE = {
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
        self.module_name = specification["module_name"]
        self.layers: List[Layer] = list(map(parse_layer, specification["layers"]))
        self.input_dim, self.output_dim = Model._calculate_tensor_shapes(self.layers)

    @staticmethod
    def _calculate_tensor_shapes(layers: list[Layer]) -> tuple[int, int]:
        """
        Calculates the shapes a sample input would have when going through the model.

        This also validates that all input/output dimensions of the layers fit
        together (e.g. we do not use a 2d convolution after a flatten layer),
        and raises an error if this is not the case.

        Returns the total input/output shapes of the model.
        """
        if len(layers) == 0:
            raise ValueError("Empty model from specification.")
        x_dim = layers[0].input_dim
        if x_dim == -1:
            raise ValueError(
                f"Model shall not start with a flexible input-size layer, found: {layers[0]}"
            )

        for l in layers:
            if l.input_dim == -1 or x_dim == l.input_dim:
                if l.output_dim == -1:
                    pass  # x_dim simply stays the way it is
                else:
                    x_dim = l.output_dim

            if l.input_dim != -1 and x_dim != l.input_dim:
                raise ValueError(
                    f"Model had input sizes missmatch at {l}, expected dim {l.input_dim}, found {x_dim}."
                )
            assert x_dim != -1  # dummy check
        return layers[0].input_dim, x_dim

import ast
from typing import Optional
from ._interfaces import Layer, Weight


class LinearLayer(Layer):
    """The abstract base class for all layers. An object of this class represents
    a layer that can be parsed from a specification and be transformed into both
    Rust and Python code.
    """

    def __init__(self, spec):
        """
        Base initialization for all layers. 
        The name is always read from the specification in the same way.
        """
        super().__init__(spec)
        self.in_channels = spec["in_channels"]
        self.out_channels = spec["out_channels"]
        self.bias = spec["bias"]
        self._name = spec["name"]


    @property
    def name(self) -> str:
        """Returns the name of the layer."""
        return self._name

    @property
    def type_py(self) -> str:
        return "Linear"

    @property
    def type_rust(self) -> str:
        return "LinearLayer"

    @property
    def weights(self) -> list[Optional[Weight]]:
        kernel = Weight("weight", (self.in_channels, self.out_channels))
        if self.bias is False:
            bias = None
        else:
            bias = Weight("bias", (self.out_channels,), optional=True)
        return [kernel, bias]

    @property
    def args_py(self) -> dict[str, str]:
        return {
            "in_features": str(self.in_channels),
            "out_features": str(self.out_channels),            
            "bias": str(self.bias)
        }

    @property
    def args_rust(self) -> list[str]:
        return []
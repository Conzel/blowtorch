import ast
from typing import Optional
from ._interfaces import Layer, Weight


class LinearLayer(Layer):
    """Base class for the linear layer. Represents the linear layer that can be
    rendered with both python and rust.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.in_features = spec["in_features"]
        self.out_features = spec["out_features"]
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
        kernel = Weight("weight", (self.out_features, self.in_features))
        if self.bias is False:
            bias = None
        else:
            bias = Weight("bias", (self.out_features,), optional=True)
        return [kernel, bias]

    @property
    def args_py(self) -> dict[str, str]:
        return {
            "in_features": str(self.in_features),
            "out_features": str(self.out_features),
            "bias": str(self.bias),
        }

    @property
    def args_rust(self) -> list[str]:
        return []

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1

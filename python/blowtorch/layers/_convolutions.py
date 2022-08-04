import ast
from typing import Optional
from ._interfaces import Layer, Weight


class Conv2dBase(Layer):
    """
    Base class for 2d convolutions, as there is not much change between
    Conv2d and Conv2dTranspose.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.in_channels = spec["in_channels"]
        self.out_channels = spec["out_channels"]
        self.kernel_size = ast.literal_eval(spec["kernel_size"])
        self.stride = spec.get("stride", 1)
        self.padding = spec.get("padding", "valid")
        self.bias = spec["bias"]

    @property
    def args_py(self) -> dict[str, str]:
        if self.padding == "valid":
            padding = "0"
        elif self.padding == "same":
            if self.kernel_size[0] != self.kernel_size[1]:
                raise ValueError(
                    "Not supported: padding=same and kernel_size[0] != kernel_size[1]"
                )
            padding = str(self.kernel_size[0] // 2)
        else:
            raise ValueError(f"Unknown padding {self.padding}")

        return {
            "in_channels": str(self.in_channels),
            "out_channels": str(self.out_channels),
            "kernel_size": str(self.kernel_size),
            "stride": str(self.stride),
            "padding": padding,
            "bias": str(self.bias),
        }

    @property
    def weights(self) -> list[Optional[Weight]]:
        kernel = Weight(
            "weight",
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
        )
        if self.bias is False:
            bias = None
        else:
            bias = Weight("bias", (self.out_channels,), optional=True)
        return [kernel, bias]

    @property
    def args_rust(self) -> list[str]:
        return [self.stride, parse_padding_from_string(self.padding)]

    @property
    def input_dim(self) -> int:
        return 3

    @property
    def output_dim(self) -> int:
        return 3


class Conv2d(Conv2dBase):
    """
    Represents a 2d convolutional layer that can be rendered in Python and Rust.
    """
    @property
    def type_py(self) -> str:
        return "Conv2d"

    @property
    def type_rust(self) -> str:
        return "ConvolutionLayer"


class Conv2dTranspose(Conv2dBase):
    """Represents a 2d transposed convolutional layer that can be rendered in Python and Rust."""

    @property
    def type_py(self) -> str:
        return "ConvTranspose2d"

    @property
    def type_rust(self) -> str:
        return "TransposedConvolutionLayer"


def parse_padding_from_string(padding: str) -> str:
    """Returns Rust padding from a padding string in {same, valid}."""
    assert padding is not None
    if padding.lower() == "same":
        return "Padding::Same"
    elif padding.lower() == "valid":
        return "Padding::Valid"
    else:
        raise ValueError(f"Unknown padding {padding}")

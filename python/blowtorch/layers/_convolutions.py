import ast
from typing import Optional
from ._interfaces import Layer, Weight


class Conv2dBase(Layer):
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
                    "Not supported: padding=same and kernel_size[0] != kernel_size[1]")
            padding = str(self.kernel_size[0] // 2)
        else:
            raise ValueError(f"Unknown padding {self.padding}")

        return {
            "in_channels": str(self.in_channels),
            "out_channels": str(self.out_channels),
            "kernel_size": str(self.kernel_size),
            "stride": str(self.stride),
            "padding": padding,
            "bias": str(self.bias)
        }

    @property
    def weights(self) -> list[Optional[Weight]]:
        # TODO: Does weight need to be transferred to both children?
        kernel = Weight("weight", (self.out_channels, self.in_channels,
                        self.kernel_size[0], self.kernel_size[1]))
        if self.bias is False:
            bias = None
        else:
            bias = Weight("bias", (self.out_channels,), optional=True)
        return [kernel, bias]

    @property
    def args_rust(self) -> list[str]:
        return [self.stride, parse_padding_from_string(self.padding)]


class Conv2d(Conv2dBase):
    @property
    def type_py(self) -> str:
        return "Conv2d"

    @property
    def type_rust(self) -> str:
        return "ConvolutionLayer"


class Conv2dTranspose(Conv2dBase):
    """TODO"""
    @property
    def type_py(self) -> str:
        return "ConvTranspose2d"

    @property
    def type_rust(self) -> str:
        return "TransposedConvolutionLayer"


def parse_padding_from_string(padding: str) -> str:
    assert padding is not None
    if padding.lower() == "same":
        return "Padding::Same"
    elif padding.lower() == "valid":
        return "Padding::Valid"
    else:
        raise ValueError(f"Unknown padding {padding}")

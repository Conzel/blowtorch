#!/usr/bin/env python3
from __future__ import annotations
import ast
from pathlib import Path
from typing import Optional
import jinja2
import os
import json
import argparse
from abc import ABC, abstractmethod

# classes


class Weight():
    """TODO"""

    def __init__(self, name: str, shape: tuple[int, ...], optional: bool = False) -> None:
        """TODO"""
        self.name = name
        self.shape = str(shape)
        self.optional = optional


class Layer(ABC):
    """TODO"""

    def __init__(self, spec: dict):
        """TODO"""
        self._name = spec["name"]

    @property
    def name(self) -> str:
        """TODO"""
        return self._name

    @property
    @abstractmethod
    def type_py(self) -> str:
        """TODO"""
        pass

    @property
    @abstractmethod
    def args_py(self) -> dict[str, str]:
        """TODO"""
        pass

    @property
    @abstractmethod
    def type_rust(self) -> str:
        """TODO"""
        pass

    @property
    @abstractmethod
    def weights(self) -> list[Optional[Weight]]:
        """TODO"""
        pass

    @property
    @abstractmethod
    def args_rust(self) -> list[str]:
        """TODO"""
        pass


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


class Relu(Layer):
    """TODO"""

    def __init__(self, spec: dict) -> None:
        """TODO"""
        super().__init__(spec)
        pass

    @property
    def type_py(self) -> str:
        return "ReLU"

    @property
    def args_py(self) -> dict[str, str]:
        return {}

    @property
    def type_rust(self) -> str:
        return "ReluLayer"

    @property
    def weights(self) -> list[Weight]:
        return []

    @property
    def args_rust(self) -> list[str]:
        return []


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

# rendering stuff


def get_template(name: str):
    template_path = Path(__file__).parent / "templates"
    loader = jinja2.FileSystemLoader(template_path)
    env = jinja2.Environment(loader=loader)
    return env.get_template(name)


def write_output(filename: str, content: str):
    with open(filename, "w+") as output_file:
        output_file.write(content)
        print(f"Successfully wrote output to {filename}")


def make_py(models: list[Model], debug: bool = False):
    template = get_template("models_template.py.jinja2")

    content = template.render(
        models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.py")
    write_output(model_output_file, content)


def make_rs(models: list[Model], debug: bool = False):
    template = get_template("models_template.rs.jinja2")

    content = template.render(
        models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.rs")
    write_output(model_output_file, content)


def models_from_spec(spec: str) -> list[Model]:
    specification_file = open(spec, "r")
    specifications = json.load(specification_file)
    return list(map(Model, specifications))


def generate_models(spec: str, debug: bool = False):
    models = models_from_spec(spec)
    make_py(models, debug)
    make_rs(models, debug)


if __name__ == "__main__":
    main()

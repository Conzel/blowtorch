#!/usr/bin/env python3
from __future__ import annotations
import ast
import pathlib
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
    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader)
    return env.get_template(name)


def write_output(filename: str, content: str):
    with open(filename, "w+") as output_file:
        output_file.write(content)
        print(f"Successfully wrote output to {filename}")


def make_py(models: list[Model]):
    template = get_template("models_template.py.jinja2")

    content = template.render(
        models=models, file=__file__, debug=args.debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.py")
    write_output(model_output_file, content)


def make_rs(models: list[Model]):
    template = get_template("models_template.rs.jinja2")

    content = template.render(
        models=models, file=__file__, debug=args.debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.rs")
    write_output(model_output_file, content)


def make_export(models: list[Model]):
    template = get_template("export_weights.py.jinja2")
    keys_to_export = []
    for model in models:
        for layer in model.layers:
            for weight in layer.weights:
                if weight is not None:
                    keys_to_export.append(
                        f"{model.module_name}.{layer.name}.{weight.name}")

    content = template.render(
        keys=keys_to_export, file=__file__, debug=args.debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "export_weights.py")
    write_output(model_output_file, content)


def main(args):
    specification_file = open(args.specification, "r")
    specifications = json.load(specification_file)
    models = list(map(Model, specifications))
    make_py(models)
    make_rs(models)
    make_export(models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Render a model file from a specification.')
    parser.add_argument('specification', metavar='SPEC', type=pathlib.Path,
                        help='Specification we should use to create the model file.')
    parser.add_argument("--debug", action='store_true',
                        help='Debug mode. Will activate trace outputs in the model output file.')

    args = parser.parse_args()

    main(args)

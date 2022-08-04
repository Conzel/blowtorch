from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod


class Weight:
    """
    Base class for a weight that a layer can have. This weight
    will be used to load the weight from the npz weights file and
    thus be rendered in Rust.

    For description of attributes, see init method.

    Attributes:
        name: Name of the weight.
        shape: Shape of the weight.
        optional: Whether the weight is optional.
    """

    def __init__(
        self, name: str, shape: tuple[int, ...], optional: bool = False
    ) -> None:
        """Initializes a layer weight

        Args:
            name: Name of the weight. This must be equal to the name of the struct
                used in Rust to represent the weight.
            shape: Shape of the weight. This is used in the Rust code to load the weight from
                the npz weights file.
            optional: Whether the weight is optional. If it is optional, the Rust code will
                wrap its type in an optional type.

        """
        self.name = name
        self.shape = str(shape)
        self.optional = optional


class Layer(ABC):
    """The abstract base class for all layers. An object of this class represents
    a layer that can be parsed from a specification and be transformed into both
    Rust and Python code.

    When implementing a new layer, just replace the abstract methods with appropriate
    parts.
    """

    def __init__(self, spec: dict):
        """
        Base initialization for all layers.
        The name is always read from the specification in the same way.
        """
        self._name = spec["name"]

    @property
    def name(self) -> str:
        """Returns the name of the layer."""
        return self._name

    def __str__(self) -> str:
        return f"Layer[{self.name}]"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    @abstractmethod
    def type_py(self) -> str:
        """Name of the Python class that represents the layer."""
        pass

    @property
    @abstractmethod
    def args_py(self) -> dict[str, str]:
        """Arguments (keyword-args) that are passed to the constructor
        of this layer."""
        pass

    @property
    @abstractmethod
    def type_rust(self) -> str:
        """Name of the Rust class that represents the layer."""
        pass

    @property
    @abstractmethod
    def weights(self) -> list[Optional[Weight]]:
        """List of weights that are used by this layer. If a layer weight is absent
        (for example if the layer does not have a bias), the corresponding element is set to None."""
        pass

    @property
    @abstractmethod
    def args_rust(self) -> list[str]:
        """List of arguments that are passed to the constructor of the Rust layer (besides weights,
        which are specified in the weights property). Has to be in the correct order."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        Returns expected dimension of the input. -1 stands for a
        wildcard, meaning the layer accepts any input.
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns output dimension of the layer. -1 stands for a wildcard, meaning
        that the output dimension is the same as the input dimension.
        """
        pass

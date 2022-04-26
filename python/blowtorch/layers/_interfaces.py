from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod


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

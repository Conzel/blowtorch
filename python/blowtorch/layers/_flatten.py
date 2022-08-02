from ._interfaces import Weight, Layer


class Flatten(Layer):
    """Base class for flatten layer that flattens array to a vector.
    Takes input array and flattens it to a vector.
    """

    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        pass

    @property
    def type_py(self) -> str:
        return "Flatten"

    @property
    def args_py(self) -> dict[str, str]:
        return {}

    @property
    def type_rust(self) -> str:
        return "Flatten"

    @property
    def weights(self) -> list[Weight]:
        return []

    @property
    def args_rust(self) -> list[str]:
        return []

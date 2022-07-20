from ._interfaces import Weight, Layer


class Flatten(Layer):
    """ReLU activation function that can be rendered in Python and Rust."""

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
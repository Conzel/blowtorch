from ._interfaces import Weight, Layer


class Relu(Layer):
    """ReLU activation function that can be rendered in Python and Rust."""

    def __init__(self, spec: dict) -> None:
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

    @property
    def input_dim(self) -> int:
        return -1

    @property
    def output_dim(self) -> int:
        return -1

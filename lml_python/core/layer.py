from tensor import Tensor
from lml_python.core.tmath import (
    matmul,
    matadd
)
from lml_python.core.interfaces import (
    ILayer,
    ITensor,
)


class Linear(ILayer):
    _weights: ITensor
    _bias: ITensor

    def __init__(self, input_size: int, output_size: int):
        self._weights = Tensor.with_uniform(
            (input_size, output_size), (-1.0, 1.0))
        self._bias = Tensor.with_uniform((output_size,), (-1.0, 1.0))

    def forward(self, input: ITensor) -> ITensor:
        out = Tensor.with_zeros((input.shape[0], self._bias.shape[0]))
        return matadd(matmul(input, self._weights, out), self._bias, out)

    def backward(self, gradient: ITensor) -> ITensor:
        pass

    def update(self, lr: float):
        pass

class ReLU(ILayer):
    def forward(self, input: ITensor) -> ITensor:
        pass

    def backward(self, gradient: ITensor) -> ITensor:
        pass

    def update(self, lr: float):
        pass
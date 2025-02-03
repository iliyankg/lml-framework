import random
import itertools
import math
import lml_python.core.vmath as vmath
from lml_python.core.tmath import matmul
from lml_python.core.interfaces import (
    TensorData,
    TensorShape,
    TensorTarget,
    ITensor,
)


class Tensor(ITensor):
    _data: TensorData
    _shape: TensorShape
    _rank: int
    _strides: list[int]

    def __init__(self, data: TensorData, shape: TensorShape):
        """Base constructor for a Tensor

        Infers strides for row major order by default
        Infers rank from inbound shape

        Args:
            data (TensorData): Raw tensor data
            shape (TensorShape): Shape of the tensor. Used to infer rank and strides
        """
        # TODO: Implement dedicated viewer/iterator for a tensor
        # TODO: [C/C++] Implement dtype akin to numpy and pytorch

        self._data = data
        self._shape = shape
        self._rank = len(shape)
        self._strides = self._calculate_strides(shape)

    def __getitem__(self, key: TensorTarget) -> float:
        assert len(self.shape) == len(key)
        fi = self._flat_idx(key)
        if fi >= len(self._data):
            raise IndexError("Index out of bounds")
        return self._data[self._flat_idx(key)]

    def __setitem__(self, key: TensorTarget, value: float):
        assert len(self.shape) == len(key)
        fi = self._flat_idx(key)
        if fi >= len(self._data):
            raise IndexError("Index out of bounds")
        self._data[self._flat_idx(key)] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self._data == other._data and self.shape == other.shape

    def __repr__(self) -> str:
        return f"""Tensor(
    shape={self.shape},
    rank={self._rank},
    strides={self._strides},
    data={self._data})
"""

    @property
    def shape(self) -> TensorShape:
        return self._shape

    @property
    def data(self) -> TensorData:
        return self._data

    @property
    def rank(self) -> int:
        return self._rank

    def reshape(self, shape: TensorShape) -> 'Tensor':
        """Reshape the tensor

        Args:
            shape (TensorShape): New shape for the tensor

        Returns:
            Tensor: Newly reshaped tensor
        """
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError("New shape must have the same number of elements")

        self._shape = shape
        self._rank = len(shape)
        self._strides = self._calculate_strides(shape)
        return self

    @classmethod
    def with_list(cls, data: list, shape: TensorShape) -> 'Tensor':
        """Create a tensor from a list

        Flattening is done row first

        Args:
            data (list): Flat or nested list of data. base data type should be float
            shape (TensorShape): Shape of the tensor

        Returns:
            Tensor: Newly created trensor
        """
        flattened = data
        while isinstance(flattened[0], list):
            flattened = list(itertools.chain.from_iterable(flattened))

        return cls(flattened, shape)

    @classmethod
    def with_uniform(cls, shape: TensorShape, uniform_range: tuple[float, float], ) -> 'Tensor':
        """Create a tensor of the desired shape with random values in a uniform distribution

        Args:
            shape (TensorShape): Shape of the tensor
            uniform_range (tuple[float, float]): The range of values to generate

        Returns:
            Tensor: Newly created tensor
        """
        num_elems = math.prod(shape)
        d = [random.uniform(*uniform_range) for _ in range(num_elems)]
        return cls(d, shape)

    @classmethod
    def with_zeros(cls, shape: TensorShape) -> 'Tensor':
        """Create a tensor of the desired shape filled with zeros

        Args:
            shape (TensorShape): Shape of the tensor

        Returns:
            Tensor: Newly created tensor
        """
        d = [0.0] * math.prod(shape)
        return cls(d, shape)

    def __matmul__(self, other: ITensor) -> ITensor:
        """Multiplies two tensors together.

        Follows matrix multiplication rules

        Args:
            other (ITensor): Other tensor

        Returns:
            ITensor: Product of the two tensors
        """
        return matmul(self, other)

    def _flat_idx(self, key: TensorTarget) -> int:
        # TODO: This might not hold true for non-rectangular tensors
        return vmath.dot(key, self._strides)  # type: ignore

    def _calculate_strides(self, shape: TensorShape) -> list[int]:
        # TODO: Fairly self contained and does not really need to be a
        # member function
        num_dims = len(shape)
        strides = [1] * num_dims
        for i in range(1, num_dims):
            prev_dim = shape[num_dims - i]
            prev_stride = strides[num_dims - i]
            # reverse order the strides for row-major order
            strides[num_dims - i - 1] = prev_stride * prev_dim
        return strides

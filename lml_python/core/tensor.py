import random
import itertools
import math
import lml_python.core.vmath as vmath

type _TensorData = list[float]
type _TensorShape = tuple[int, ...]
type _TensorTarget = tuple[int, ...]


class Tensor(object):
    data: _TensorData
    shape: tuple[int, ...]
    rank: int
    strides: list[int]

    def __init__(self, data: _TensorData, shape: _TensorShape):
        """Base constructor for a Tensor

        Infers strides for row major order by default
        Infers rank from inbound shape

        Args:
            data (_TensorData): Raw tensor data
            shape (_TensorShape): Shape of the tensor. Used to infer rank and strides
        """
        # TODO: Implement dedicated viewer/iterator for a tensor
        # TODO: [C/C++] Implement dtype akin to numpy and pytorch

        self.data = data
        self.shape = shape
        self.rank = len(shape)
        self.strides = [1] * self.rank

        for i in range(1, self.rank):
            prev_dim = self.shape[self.rank-i]
            prev_stride = self.strides[self.rank-i]
            # reverse order the strides for row-major order
            self.strides[self.rank-i-1] = prev_stride * prev_dim

    def __getitem__(self, key: _TensorTarget) -> float:
        assert len(self.shape) == len(key)
        return self.data[self._flat_idx(key)]

    def __repr__(self) -> str:
        return f"""Tensor(
    shape={self.shape},
    rank={self.rank},
    strides={self.strides},
    data={self.data})
"""

    @classmethod
    def with_list(cls, data: list, shape: _TensorShape) -> 'Tensor':
        """Create a tensor from a list

        Flattening is done row first

        Args:
            data (list): Flat or nested list of data. base data type should be float
            shape (_TensorShape): Shape of the tensor

        Returns:
            Tensor: Newly created trensor
        """
        flattened = data
        while isinstance(flattened[0], list):
            flattened = list(itertools.chain.from_iterable(flattened))

        return cls(flattened, shape)

    @classmethod
    def with_uniform(cls, shape: _TensorShape, uniform_range: tuple[float, float], ) -> 'Tensor':
        """Create a tensor of the desired shape with random values in a uniform distribution

        Args:
            shape (_TensorShape): Shape of the tensor
            uniform_range (tuple[float, float]): The range of values to generate

        Returns:
            Tensor: Newly created tensor
        """
        num_elems = math.prod(shape)
        d = [random.uniform(*uniform_range) for _ in range(num_elems)]
        return cls(d, shape)

    @classmethod
    def with_zeros(cls, shape: _TensorShape) -> 'Tensor':
        """Create a tensor of the desired shape filled with zeros

        Args:
            shape (_TensorShape): Shape of the tensor

        Returns:
            Tensor: Newly created tensor
        """
        d = [0.0] * math.prod(shape)
        return cls(d, shape)

    def _flat_idx(self, key: _TensorTarget) -> int:
        # TODO: This might not hold true for non-rectangular tensors
        return vmath.dot(key, self.strides)  # type: ignore

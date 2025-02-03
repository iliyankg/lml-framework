from abc import ABC, abstractmethod

type TensorData = list[float]
type TensorShape = tuple[int, ...]
type TensorTarget = tuple[int, ...]


class ITensor(ABC):
    @abstractmethod
    def __init__(self, data: TensorData, shape: TensorShape):
        pass

    @abstractmethod
    def __getitem__(self, key: TensorTarget) -> float:
        pass

    @abstractmethod
    def __setitem__(self, key: TensorTarget, value: float):
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __matmul__(self, other: 'ITensor') -> 'ITensor':
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @property
    @abstractmethod
    def shape(self) -> TensorShape:
        pass

    @property
    @abstractmethod
    def data(self) -> TensorData:
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        pass

    @abstractmethod
    def reshape(self, shape: TensorShape) -> 'ITensor':
        pass

    @classmethod
    @abstractmethod
    def with_list(cls, data: TensorData, shape: TensorShape) -> 'ITensor':
        pass

    @classmethod
    @abstractmethod
    def with_zeros(cls, shape: TensorShape) -> 'ITensor':
        pass

    @classmethod
    @abstractmethod
    def with_uniform(cls, shape: TensorShape, low: float, high: float) -> 'ITensor':
        pass

from typing import Optional
from tensor import Tensor, _TensorShape


def tdot(a: Tensor,
         b: Tensor,
         axes: int | tuple[_TensorShape, _TensorShape],
         out: Tensor):
    # TODO: Next
    # Process the axes and ensure they are in the expected shape
    # Validate input tensor shapes are compatible
    # Calculate output shape and set up output tensor
    # Compute and set result either to output or as a return value

    raise NotImplementedError

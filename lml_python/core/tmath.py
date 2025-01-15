from typing import Optional
from lml_python.core.tensor import Tensor, _TensorShape, _TensorData
import lml_python.core.vmath as vmath


def matmul(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Simple 2D matrix multiplication

    Demands 1D or 2D tensors (matrices), can optionally be provided with an
    output tensor.

    Note: For 1D tensors the shape currently must be (1, N) or (N, 1).
    TODO: Implement support for 1D tensors with shape (N,)

    Args:
         a (Tensor): Left 2D tensor to multiply
         b (Tensor): Right 2D tensor to multiply
         out (Optional[Tensor], optional): Output tensor to store the 
            result in. At present out tensor shape and data length must
            be aligned with the expected output shape and data length.
            Defaults to None.

    Returns:
         Tensor: Tensor containing the result of the matrix multiplication.
         If 'out' is provided it is mutated and also returned.
    """

    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both tensors must be 2D")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shapes {a.shape} and {b.shape} are not aligned")

    if out is not None:
        if out.shape != (a.shape[0], b.shape[1]):
            raise ValueError("Output tensor shape is not aligned")
        if len(out.data) != a.shape[0] * b.shape[1]:
            raise ValueError("Output tensor data length is not aligned")
    else:
        out = Tensor.with_zeros(shape=(a.shape[0], b.shape[1]))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(b.shape[1]):
                out[i, k] += a[i, j] * b[j, k]

    return out


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

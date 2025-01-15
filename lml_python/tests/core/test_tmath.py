import pytest
from lml_python.core.tensor import Tensor
from lml_python.core.tmath import matmul


@pytest.mark.parametrize("a, b, expected", [
    (Tensor.with_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3)),
     Tensor.with_list([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (3, 2)),
     Tensor.with_list([[22.0, 28.0], [49.0, 64.0]], (2, 2))),
    (Tensor.with_list([[1.0, 2.0], [3.0, 4.0]], (2, 2)),
     Tensor.with_list([[1.0, 2.0], [3.0, 4.0]], (2, 2)),
     Tensor.with_list([[7.0, 10.0], [15.0, 22.0]], (2, 2))),
    (Tensor.with_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], (3, 3)),
     Tensor.with_list([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (3, 2)),
     Tensor.with_list([[22.0, 28.0], [49.0, 64.0], [76.0, 100.0]], (3, 2))),
    (Tensor.with_list([1.0, 2.0, 3.0], (1, 3)),
     Tensor.with_list([1.0, 2.0, 3.0], (3, 1)),
     Tensor.with_list([14], (1, 1))),
])
def test_matmul(a, b, expected):
    out = Tensor.with_zeros((a.shape[0], b.shape[1]))
    assert expected == matmul(a, b, out)
    assert expected == out
    assert expected == matmul(a, b)
    

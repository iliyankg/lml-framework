import pytest
import math
from lml_python.core.tensor import Tensor

# TODO: Restructure this fixtures for redability, re-use and parametrization


@pytest.fixture(params=[
    ([1.0, 2.0, 3.0],
     (3,), 1, [1]),

    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
     (2, 3), 2, [3, 1]),

    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
      19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
     (3, 3, 3), 3, [9, 3, 1]),
],
    ids=["1D", "2D", "3D"])
def raw_tensor_data(request: pytest.FixtureRequest):
    return request.param


def test_default_constructor_initialization(raw_tensor_data):
    data, shape, rank, strides = raw_tensor_data

    tensor = Tensor(data, shape)
    assert tensor.data == data
    assert tensor.shape == shape
    assert tensor.rank == rank
    assert tensor.strides == strides


@pytest.fixture(params=[
    ([1.0, 2.0, 3.0], (3,), 1, [1], [1.0, 2.0, 3.0], (0,), 1.0),

    ([1.0, 2.0, 3.0], (3,), 1, [1], [1.0, 2.0, 3.0], (1,), 2.0),

    ([1.0, 2.0, 3.0], (3,), 1, [1], [1.0, 2.0, 3.0], (2,), 3.0),

    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3), 2, [3, 1],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (0, 0,), 1.0),

    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3), 2, [3, 1],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (0, 2,), 3.0),

    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3), 2, [3, 1],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 1,), 5.0),

    ([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
      [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
      [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]]],
     (3, 3, 3), 3, [9, 3, 1],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
      19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], (1, 2, 2,), 18.0),

    ([[[[[1.0], [2.0]], [[3.0], [4.0]]]]],
     (1, 1, 2, 2, 1), 5, [4, 4, 2, 1, 1],
     [1.0, 2.0, 3.0, 4.0], (0, 0, 0, 1, 0,), 2.0),

    ([[[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]],
       [[4.0, 4.5], [5.0, 5.5], [6.0, 6.5]],
       [[7.0, 7.5], [8.0, 8.5], [9.0, 9.5]]],
      [[[10.0, 10.5], [11.0, 11.5], [12.0, 12.5]],
       [[13.0, 13.5], [14.0, 14.5], [15.0, 15.5]],
       [[16.0, 16.5], [17.0, 17.5], [18.0, 18.5]]],
      [[[19.0, 19.5], [20.0, 20.5], [21.0, 21.5]],
       [[22.0, 22.5], [23.0, 23.5], [24.0, 24.5]],
       [[25.0, 25.5], [26.0, 26.5], [27.0, 27.5]]],
      ], (3, 3, 3, 2), 4, [18, 6, 2, 1],
     [1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
      4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
      7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
      10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
      13.0, 13.5, 14.0, 14.5, 15.0, 15.5,
      16.0, 16.5, 17.0, 17.5, 18.0, 18.5,
      19.0, 19.5, 20.0, 20.5, 21.0, 21.5,
      22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
      25.0, 25.5, 26.0, 26.5, 27.0, 27.5],
     (1, 2, 1, 1,), 17.5)],
    ids=["1D_0", "1D_1", "1D_2", "2D_0", "2D_1", "2D_2", "3D_0", "5D_0", "5D_1"])
def tensor_data(request):
    return request.param


def test_tensor_with_iterable(tensor_data):
    data, shape, rank, strides, expected_raw_data, _, _ = tensor_data
    tensor = Tensor.with_list(data, shape)
    assert tensor.rank == rank
    assert tensor.strides == strides
    assert tensor.data == expected_raw_data
    assert tensor.shape == shape


def test_tensor_getitem(tensor_data):
    data, shape, _, _, _, indices, expected = tensor_data
    tensor = Tensor.with_list(data, shape)
    assert tensor[*indices] == expected


def test_tensor_with_zeros():
    shape = (2, 3)
    tensor = Tensor.with_zeros(shape)
    expected_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert tensor.data == expected_data
    assert tensor.shape == shape
    assert tensor.rank == 2
    assert tensor.strides == [3, 1]


@pytest.mark.parametrize("shape, uniform_range", [
    ((2, 2), (1.0, 1.0)),
    ((3, 1), (-5.0, 5.0)),
    ((1, 3), (-2.0, -2.0)),
    ((2, 3), (0.0, 0.0)),
])
def test_tensor_with_uniform(shape, uniform_range):
    tensor = Tensor.with_uniform(shape, uniform_range)
    assert tensor.shape == shape
    # Assuming the tensor data is generated within the uniform range
    assert all(uniform_range[0] <= value <= uniform_range[1]
               for value in tensor.data)


@pytest.mark.parametrize("initial_shape, new_shape, old_index, new_index", [
    ((2, 3), (3, 2), (0, 2), (1, 0)),
    ((3, 3), (1, 9), (1, 1), (0, 4)),
    ((4, 1), (2, 2), (2, 0), (1, 0)),
    ((2, 2, 2), (4, 2), (1, 0, 1), (2, 1)),
])
def test_tensor_reshape_various_shapes(initial_shape,
                                       new_shape,
                                       old_index,
                                       new_index):
    tensor = Tensor.with_uniform(initial_shape, (0.0, 1.0))
    val = tensor[old_index]
    reshaped_tensor = tensor.reshape(new_shape)
    reshaped_val = reshaped_tensor[new_index]
    assert reshaped_val == val
    assert reshaped_tensor.shape == new_shape
    assert reshaped_tensor.data == tensor.data
    assert math.prod(reshaped_tensor.shape) == math.prod(initial_shape)

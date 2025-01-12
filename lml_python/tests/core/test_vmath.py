import pytest
from lml_python.core.vmath import dot


@pytest.mark.parametrize("vec1, vec2, expected", [
    ([1, 2, 3], [4, 5, 6], 32),
    ([0, 0, 0], [1, 2, 3], 0),
    ([1.5, 2.5], [3.5, 4.5], 16.5),
    ([-1, -2, -3], [-4, -5, -6], 32),
    ([1, 2], [3, 4], 11),
    ((1, 2, 3), (4, 5, 6), 32),
    ((0, 0, 0), (1, 2, 3), 0),
    ((1.5, 2.5), (3.5, 4.5), 16.5),
    ((-1, -2, -3), (-4, -5, -6), 32),
    ((1, 2), (3, 4), 11),
])
def test_dot_product(vec1, vec2, expected):
    assert dot(vec1, vec2) == expected


def test_dot_product_unequal_length():
    with pytest.raises(AssertionError):
        dot([1, 2], [1, 2, 3])

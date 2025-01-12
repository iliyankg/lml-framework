type _ListLike = list[int] | tuple[int, ...] | list[float] | tuple[float, ...]


def dot(a: _ListLike, b: _ListLike) -> float | int:
    assert len(a) == len(b), "Dot Product requires vectors of equal length"
    return sum(x*y for x, y in zip(a, b))

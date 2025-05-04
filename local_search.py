from typing import List, Sequence

def _d(a: int, b: int, dist: Sequence[Sequence[float]]) -> float:
    return dist[a][b]

def cycle_cost(cycle: Sequence[int], dist: Sequence[Sequence[float]]) -> float:
    n = len(cycle)
    return sum(_d(cycle[i], cycle[(i + 1) % n], dist) for i in range(n))

def total_cost(
    c1: Sequence[int],
    c2: Sequence[int],
    dist: Sequence[Sequence[float]],
) -> float:
    return cycle_cost(c1, dist) + cycle_cost(c2, dist)

def delta_2opt(
    cycle: Sequence[int],
    i: int,
    j: int,
    dist: Sequence[Sequence[float]],
) -> float:
    n = len(cycle)
    a, b = cycle[i], cycle[(i + 1) % n]
    c, d = cycle[j], cycle[(j + 1) % n]
    old = _d(a, b, dist) + _d(c, d, dist)
    new = _d(a, c, dist) + _d(b, d, dist)
    return new - old

def delta_v_between(
    c1: Sequence[int],
    c2: Sequence[int],
    i: int,
    j: int,
    dist: Sequence[Sequence[float]],
) -> float:
    n1, n2 = len(c1), len(c2)
    a_prev, a, a_next = c1[(i - 1) % n1], c1[i], c1[(i + 1) % n1]
    b_prev, b, b_next = c2[(j - 1) % n2], c2[j], c2[(j + 1) % n2]

    old = (
        _d(a_prev, a, dist)
        + _d(a, a_next, dist)
        + _d(b_prev, b, dist)
        + _d(b, b_next, dist)
    )
    new = (
        _d(a_prev, b, dist)
        + _d(b, a_next, dist)
        + _d(b_prev, a, dist)
        + _d(a, b_next, dist)
    )
    return new - old

def apply_2opt(cycle: List[int], i: int, j: int) -> List[int]:
    return cycle[: i+1] + list(reversed(cycle[i+1 : j+1])) + cycle[j+1 :]

def apply_v_between(c1: List[int], c2: List[int], i: int, j: int):
    c1[i], c2[j] = c2[j], c1[i]

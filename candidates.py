from typing import List, Sequence
import time

from local_search import apply_2opt, apply_v_between, delta_2opt, delta_v_between


def _nearest_neighbors_and_mask(dist: Sequence[Sequence[float]], k: int = 10):
    n = len(dist)
    nearest = [[] for _ in range(n)]
    mask = [[False] * n for _ in range(n)]
    for i in range(n):
        order = sorted(((j, dist[i][j]) for j in range(n) if j != i), key=lambda x: x[1])
        for j, _ in order[:k]:
            nearest[i].append(j)
            mask[i][j] = True
    return nearest, mask

def local_search_with_candidates(
    cycle1: List[int],
    cycle2: List[int],
    dist: Sequence[Sequence[float]],
    k: int = 10,
):
    start = time.time()

    cycles = [cycle1[:], cycle2[:]]
    nearest, nn_mask = _nearest_neighbors_and_mask(dist, k)

    while True:
        best = (0.0, None, None)  # (delta, move, type)
        positions = [{v: i for i, v in enumerate(c)} for c in cycles]

        # ---------- 1. 2‑opt inside each cycle ----------
        for idx, c in enumerate(cycles):
            n = len(c)
            pos = positions[idx]
            for i, v_i in enumerate(c):
                v_i_next = c[(i + 1) % n]
                for v_j in nearest[v_i]:
                    j = pos.get(v_j)
                    if j is None or j <= i + 1 or j >= n - 1:
                        continue
                    v_j_next = c[(j + 1) % n]
                    if not (
                        nn_mask[v_i_next][v_j_next]
                        or nn_mask[v_j_next][v_i_next]
                        or nn_mask[v_j][v_i]
                    ):
                        continue
                    d = delta_2opt(c, i, j, dist)
                    if d < best[0]:
                        best = (d, (idx, i, j), "2opt")

        # ---------- 2. Swap between cycles -------------
        n1, n2 = map(len, cycles)
        for i, v_i in enumerate(cycles[0]):
            i_prev, i_next = cycles[0][(i - 1) % n1], cycles[0][(i + 1) % n1]
            for j, v_j in enumerate(cycles[1]):
                j_prev, j_next = cycles[1][(j - 1) % n2], cycles[1][(j + 1) % n2]
                if not (
                    nn_mask[v_j][i_prev]
                    or nn_mask[v_j][i_next]
                    or nn_mask[v_i][j_prev]
                    or nn_mask[v_i][j_next]
                    or nn_mask[i_prev][v_j]
                    or nn_mask[i_next][v_j]
                    or nn_mask[j_prev][v_i]
                    or nn_mask[j_next][v_i]
                ):
                    continue
                d = delta_v_between(cycles[0], cycles[1], i, j, dist)
                if d < best[0]:
                    best = (d, (i, j), "swap")

        delta, move, mtype = best
        if delta >= 0:
            break
        if mtype == "2opt":
            idx, i, j = move
            cycles[idx] = apply_2opt(cycles[idx], i, j)
        else:  # swap
            i, j = move
            apply_v_between(cycles[0], cycles[1], i, j)

    return cycles[0], cycles[1], time.time() - start

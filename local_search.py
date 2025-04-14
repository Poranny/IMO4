import math
import time

def cycle_cost(cycle, coords):
    cost = 0.0
    n = len(cycle)
    for i in range(n):
        j = (i + 1) % n
        cost += eucl_dist(coords[cycle[i]], coords[cycle[j]])
    return cost

def total_cost(cycle1, cycle2, coords):
    return cycle_cost(cycle1, coords) + cycle_cost(cycle2, coords)

def eucl_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

def delta_2opt(cycle, coords, i, j):
    n = len(cycle)

    i_next = (i + 1) % n
    j_next = (j + 1) % n

    old_dist = (eucl_dist(coords[cycle[i]], coords[cycle[i_next]]) +
                eucl_dist(coords[cycle[j]], coords[cycle[j_next]]))

    new_dist = (eucl_dist(coords[cycle[i]], coords[cycle[j]]) +
                eucl_dist(coords[cycle[i_next]], coords[cycle[j_next]]))

    return new_dist - old_dist


def apply_2opt(cycle, i, j):
    return cycle[:i + 1] + list(reversed(cycle[i + 1:j + 1])) + cycle[j + 1:]

def delta_between(c1, c2, coords, i, j):
    n1 = len(c1)
    n2 = len(c2)

    i_prev = (i - 1) % n1
    i_next = (i + 1) % n1
    j_prev = (j - 1) % n2
    j_next = (j + 1) % n2

    old_cost = (
            eucl_dist(coords[c1[i_prev]], coords[c1[i]]) +
            eucl_dist(coords[c1[i]], coords[c1[i_next]]) +
            eucl_dist(coords[c2[j_prev]], coords[c2[j]]) +
            eucl_dist(coords[c2[j]], coords[c2[j_next]])
    )

    new_cost = (
            eucl_dist(coords[c1[i_prev]], coords[c2[j]]) +
            eucl_dist(coords[c2[j]], coords[c1[i_next]]) +
            eucl_dist(coords[c2[j_prev]], coords[c1[i]]) +
            eucl_dist(coords[c1[i]], coords[c2[j_next]])
    )

    return new_cost - old_cost

def apply_between(c1, c2, i, j):
    c1[i], c2[j] = c2[j], c1[i]

def local_steepest_edges_with_inter(cycle1, cycle2, coords):
    start_time = time.time()

    c1 = cycle1[:]
    c2 = cycle2[:]

    while True:
        best_delta = 0.0
        best_move = None
        best_type = None

        n1 = len(c1)
        for i in range(n1):
            for j in range(i + 2, n1):
                if (j + 1) % n1 == i:
                    continue
                d = delta_2opt(c1, coords, i, j)
                if d < best_delta:
                    best_delta = d
                    best_move = (i, j)
                    best_type = "2opt1"

        n2 = len(c2)
        for i in range(n2):
            for j in range(i + 2, n2):
                if (j + 1) % n2 == i:
                    continue
                d = delta_2opt(c2, coords, i, j)
                if d < best_delta:
                    best_delta = d
                    best_move = (i, j)
                    best_type = "2opt2"

        for i in range(n1):
            for j in range(n2):
                d = delta_between(c1, c2, coords, i, j)
                if d < best_delta:
                    best_delta = d
                    best_move = (i, j)
                    best_type = "swap"

        if best_delta >= 0:
            break

        if best_type == "2opt1":
            i, j = best_move
            c1 = apply_2opt(c1, i, j)

        elif best_type == "2opt2":
            i, j = best_move
            c2 = apply_2opt(c2, i, j)

        elif best_type == "swap":
            i, j = best_move
            apply_between(c1, c2, i, j)

    total_t = time.time() - start_time
    return c1, c2, total_t
import tsplib95
import random
import math


def load_coords(tsp_file):
    problem = tsplib95.load(tsp_file)
    coords = [coord for _, coord in sorted(problem.node_coords.items())]
    return coords


def compute_distance_matrix(coords):
    n = len(coords)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            dm[i][j] = dist
            dm[j][i] = dist
    return dm


def generate_random_solution(coords):

    n = len(coords)
    if n < 2:
        return [], []

    indices = list(range(n))
    random.shuffle(indices)
    half = n // 2
    cycle1 = indices[:half]
    cycle2 = indices[half:]

    return cycle1, cycle2


def generate_random_two_cycles(coords):
    n = len(coords)
    indices = list(range(n))
    random.shuffle(indices)
    half = n // 2
    return indices[:half], indices[half:]


def cycle_cost(distance_matrix, cycle):
    cost = 0.0
    for i in range(len(cycle)):
        j = (i + 1) % len(cycle)
        cost += distance_matrix[cycle[i]][cycle[j]]
    return cost


def generate_weighted_regret(coords, alpha=0.75):
    distance_matrix = compute_distance_matrix(coords)
    n = len(coords)
    if n < 2:
        return [], []

    vertices = set(range(n))

    start1 = random.choice(list(vertices))
    vertices.remove(start1)

    second1 = min(vertices, key=lambda v: distance_matrix[start1][v])
    vertices.remove(second1)
    cycle1 = [start1, second1]

    if vertices:
        start2 = max(vertices, key=lambda v: distance_matrix[start1][v])
        vertices.remove(start2)
        if vertices:
            second2 = min(vertices, key=lambda v: distance_matrix[start2][v])
            vertices.remove(second2)
            cycle2 = [start2, second2]
        else:
            cycle2 = [start2]
    else:
        cycle2 = []

    turn = 0
    while vertices:
        candidates = []
        current_cycle = cycle1 if turn == 0 else cycle2
        m = len(current_cycle)

        for v in vertices:
            costs = []
            for i in range(m):
                j = (i + 1) % m
                old_e = distance_matrix[current_cycle[i]][current_cycle[j]]
                new_e = (distance_matrix[current_cycle[i]][v] +
                         distance_matrix[v][current_cycle[j]])
                cost_inc = new_e - old_e
                costs.append((cost_inc, i + 1))

            costs.sort(key=lambda x: x[0])
            best_inc, best_pos = costs[0]
            second_inc = costs[1][0] if len(costs) > 1 else best_inc

            regret = second_inc - best_inc
            if alpha != 1.0:
                regret = alpha * regret - (1.0 - alpha) * best_inc

            candidates.append((regret, best_inc, current_cycle, best_pos, v))

        if not candidates:
            break

        best_candidate = max(candidates, key=lambda x: x[0])
        _, best_inc, chosen_cycle, pos, chosen_vertex = best_candidate

        chosen_cycle.insert(pos, chosen_vertex)
        vertices.remove(chosen_vertex)
        turn = 1 - turn

    return cycle1, cycle2

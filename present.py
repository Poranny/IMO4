# present.py

import random
import time
import os
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt

from construct import load_coords, generate_weighted_regret

from local_search import (
    total_cost,
    eucl_dist,
    delta_2opt,
    apply_2opt,
    apply_between,
    local_steepest_edges_with_inter,
    delta_between
)

from movement_list import local_search_with_move_list  # importujemy funkcję z modułu movement_list

def find_nearest_neighbors(coords, k=10):
    n = len(coords)
    nearest_neighbors = []
    for i in range(n):
        distances = [(j, ((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)**0.5)
                     for j in range(n) if j != i]
        distances.sort(key=lambda x: x[1])
        nearest = [j for j, _ in distances[:k]]
        nearest_neighbors.append(nearest)
    return nearest_neighbors

def local_search_with_candidates(cycle1, cycle2, coords, k=10):
    start_time = time.time()
    c1 = cycle1[:]
    c2 = cycle2[:]
    nearest_neighbors = find_nearest_neighbors(coords, k)
    improved = True
    best_delta = 0.0
    best_move = None
    best_type = None

    while improved:
        improved = False
        best_delta = 0.0
        best_move = None
        best_type = None

        n1 = len(c1)
        for i in range(n1):
            i_next = (i + 1) % n1
            v_i = c1[i]
            v_i_next = c1[i_next]

            for j in range(i + 2, n1 - 1):
                j_next = (j + 1) % n1
                v_j = c1[j]
                v_j_next = c1[j_next]

                if (v_j in nearest_neighbors[v_i] or
                    v_j_next in nearest_neighbors[v_i_next] or
                    v_i in nearest_neighbors[v_j] or
                    v_i_next in nearest_neighbors[v_j_next]):

                    if (j + 1) % n1 == i:
                        continue

                    d = delta_2opt(c1, coords, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_move = (i, j)
                        best_type = "2opt1"

        n2 = len(c2)
        for i in range(n2):
            i_next = (i + 1) % n2
            v_i = c2[i]
            v_i_next = c2[i_next]

            for j in range(i + 2, n2 - 1):
                j_next = (j + 1) % n2
                v_j = c2[j]
                v_j_next = c2[j_next]

                if (v_j in nearest_neighbors[v_i] or
                    v_j_next in nearest_neighbors[v_i_next] or
                    v_i in nearest_neighbors[v_j] or
                    v_i_next in nearest_neighbors[v_j_next]):

                    if (j + 1) % n2 == i:
                        continue

                    d = delta_2opt(c2, coords, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_move = (i, j)
                        best_type = "2opt2"

        for i in range(n1):
            i_prev = (i - 1) % n1
            i_next = (i + 1) % n1
            v_i = c1[i]
            v_i_prev = c1[i_prev]
            v_i_next = c1[i_next]

            for j in range(n2):
                j_prev = (j - 1) % n2
                j_next = (j + 1) % n2
                v_j = c2[j]
                v_j_prev = c2[j_prev]
                v_j_next = c2[j_next]

                if (v_j in nearest_neighbors[v_i_prev] or
                    v_j in nearest_neighbors[v_i_next] or
                    v_i in nearest_neighbors[v_j_prev] or
                    v_i in nearest_neighbors[v_j_next] or
                    v_i_prev in nearest_neighbors[v_j] or
                    v_i_next in nearest_neighbors[v_j] or
                    v_j_prev in nearest_neighbors[v_i] or
                    v_j_next in nearest_neighbors[v_i]):

                    d = delta_between(c1, c2, coords, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_move = (i, j)
                        best_type = "swap"

        if best_delta < 0:
            improved = True
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

def run_experiment(tsp_files, num_runs=100):
    results = defaultdict(lambda: defaultdict(list))
    best_solutions = {}

    for tsp_file in tsp_files:
        print(f"Processing {tsp_file}...")
        coords = load_coords(tsp_file)
        best_solutions[tsp_file] = {}

        wr_total_cost = 0
        best_wr_c1, best_wr_c2 = None, None
        best_wr_cost = float('inf')

        best_ls_c1, best_ls_c2 = None, None
        best_lsml_c1, best_lsml_c2 = None, None
        best_lsc_c1, best_lsc_c2 = None, None
        best_ls_cost = float('inf')
        best_lsml_cost = float('inf')
        best_lsc_cost = float('inf')

        for _ in range(num_runs):
            c1, c2 = generate_weighted_regret(coords, alpha=0.75)
            cost = total_cost(c1, c2, coords)
            wr_total_cost += cost

            if cost < best_wr_cost:
                best_wr_cost = cost
                best_wr_c1, best_wr_c2 = c1[:], c2[:]

            # 1. Standardowy steepest local search
            ls_c1, ls_c2, ls_time = local_steepest_edges_with_inter(c1[:], c2[:], coords)
            ls_cost = total_cost(ls_c1, ls_c2, coords)
            results[tsp_file]["standard_ls_cost"].append(ls_cost)
            results[tsp_file]["standard_ls_time"].append(ls_time)

            if ls_cost < best_ls_cost:
                best_ls_cost = ls_cost
                best_ls_c1, best_ls_c2 = ls_c1[:], ls_c2[:]

            # 2. Local search z listą ruchów (importowanej funkcji)
            lsml_c1, lsml_c2, lsml_time = local_search_with_move_list(c1[:], c2[:], coords)
            lsml_cost = total_cost(lsml_c1, lsml_c2, coords)
            results[tsp_file]["move_list_ls_cost"].append(lsml_cost)
            results[tsp_file]["move_list_ls_time"].append(lsml_time)

            if lsml_cost < best_lsml_cost:
                best_lsml_cost = lsml_cost
                best_lsml_c1, best_lsml_c2 = lsml_c1[:], lsml_c2[:]

            # 3. Local search z ruchami kandydackimi
            lsc_c1, lsc_c2, lsc_time = local_search_with_candidates(c1[:], c2[:], coords, k=10)
            lsc_cost = total_cost(lsc_c1, lsc_c2, coords)
            results[tsp_file]["candidate_ls_cost"].append(lsc_cost)
            results[tsp_file]["candidate_ls_time"].append(lsc_time)

            if lsc_cost < best_lsc_cost:
                best_lsc_cost = lsc_cost
                best_lsc_c1, best_lsc_c2 = lsc_c1[:], lsc_c2[:]

        best_solutions[tsp_file]["standard_ls"] = (best_ls_c1, best_ls_c2)
        best_solutions[tsp_file]["move_list_ls"] = (best_lsml_c1, best_lsml_c2)
        best_solutions[tsp_file]["candidate_ls"] = (best_lsc_c1, best_lsc_c2)

        results[tsp_file]["weighted_regret"].append(wr_total_cost / num_runs)
        best_solutions[tsp_file]["weighted_regret"] = (best_wr_c1, best_wr_c2)

    return results, best_solutions

def print_results(results):
    print("\n=== WYNIKI EKSPERYMENTU ===")
    for tsp_file, data in results.items():
        print(f"\nInstancja: {tsp_file}")

        print("\nWyniki dla heurystyki konstrukcyjnej (weighted regret):")
        wr_cost = data["weighted_regret"][0]
        print(f"  Średni koszt: {wr_cost:.2f}")

        print("\nWyniki dla standardowego local search:")
        ls_costs = data["standard_ls_cost"]
        ls_times = data["standard_ls_time"]
        print(f"  Średni koszt: {statistics.mean(ls_costs):.2f}")
        print(f"  Minimalny koszt: {min(ls_costs):.2f}")
        print(f"  Maksymalny koszt: {max(ls_costs):.2f}")
        print(f"  Średni czas wykonania: {statistics.mean(ls_times):.4f} s")

        print("\nWyniki dla local search z listą ruchów:")
        lsml_costs = data["move_list_ls_cost"]
        lsml_times = data["move_list_ls_time"]
        print(f"  Średni koszt: {statistics.mean(lsml_costs):.2f}")
        print(f"  Minimalny koszt: {min(lsml_costs):.2f}")
        print(f"  Maksymalny koszt: {max(lsml_costs):.2f}")
        print(f"  Średni czas wykonania: {statistics.mean(lsml_times):.4f} s")

        print("\nWyniki dla local search z ruchami kandydackimi:")
        lsc_costs = data["candidate_ls_cost"]
        lsc_times = data["candidate_ls_time"]
        print(f"  Średni koszt: {statistics.mean(lsc_costs):.2f}")
        print(f"  Minimalny koszt: {min(lsc_costs):.2f}")
        print(f"  Maksymalny koszt: {max(lsc_costs):.2f}")
        print(f"  Średni czas wykonania: {statistics.mean(lsc_times):.4f} s")

def plot_solution(ax, cycle1, cycle2, coords, title):
    x1 = [coords[node][0] for node in cycle1]
    y1 = [coords[node][1] for node in cycle1]
    x1.append(x1[0])
    y1.append(y1[0])
    ax.plot(x1, y1, linewidth=1, marker='o', markersize=3)

    x2 = [coords[node][0] for node in cycle2]
    y2 = [coords[node][1] for node in cycle2]
    x2.append(x2[0])
    y2.append(y2[0])
    ax.plot(x2, y2, linewidth=1, marker='o', markersize=3)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

def plot_all_solutions(tsp_file, solutions, coords):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Najlepsze rozwiązania dla {tsp_file}', fontsize=16)

    algorithms = {
        'Weighted Regret': 'weighted_regret',
        'Standard Local Search': 'standard_ls',
        'Local Search z listą ruchów': 'move_list_ls',
        'Local Search z kandydatami': 'candidate_ls'
    }

    idx = 0
    for title, key in algorithms.items():
        row = idx // 2
        col = idx % 2
        cycle1, cycle2 = solutions[key]
        cost = total_cost(cycle1, cycle2, coords)
        plot_solution(axs[row, col], cycle1, cycle2, coords, f'{title}\nKoszt: {cost:.2f}')
        idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_dir = 'wykresy'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.basename(tsp_file).replace('.tsp', '')
    plt.savefig(f'{output_dir}/{filename}_solutions.png', dpi=300)
    plt.close()

def main():
    tsp_files = [
        "kro/kroA200.tsp",
        "kro/kroB200.tsp",
    ]
    num_runs = 2
    results, best_solutions = run_experiment(tsp_files, num_runs)
    print_results(results)
    print("\nTworzenie wykresów najlepszych rozwiązań...")
    for tsp_file in tsp_files:
        coords = load_coords(tsp_file)
        plot_all_solutions(tsp_file, best_solutions[tsp_file], coords)
    print("Wykresy zostały zapisane w katalogu 'wykresy/'.")

if __name__ == "__main__":
    main()

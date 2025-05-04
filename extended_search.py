import random
import time
from typing import List

from candidates import local_search_with_candidates
from construct import (
    load_coords,
    generate_random_solution,
    weighted_regret_insertion_balanced,
    compute_distance_matrix,
)
from local_search import total_cost
from present import plot_extended_solutions


def local_search_solution(c1: List[int], c2: List[int], dist):
    new_c1, new_c2, _ = local_search_with_candidates(c1, c2, dist, 30)
    return new_c1, new_c2

def msls(coords, dist, num_iterations: int = 200):
    best_sol = None
    best_cost = float("inf")
    start_time = time.time()

    for _ in range(num_iterations):
        c1, c2 = generate_random_solution(coords)
        ls_c1, ls_c2 = local_search_solution(c1, c2, dist)
        cost = total_cost(ls_c1, ls_c2, dist) # maybe this is bad
        if cost < best_cost:
            best_cost = cost
            best_sol = (ls_c1, ls_c2)

    return best_sol, best_cost, time.time() - start_time


# ---------------- Iterated Local Search ------------------------------------
def perturbation_ils(c1, c2, intensity: int = 1):

    new_c1, new_c2 = c1.copy(), c2.copy()

    use_first = random.random() < 0.5
    tgt = new_c1 if use_first else new_c2
    n = len(tgt)
    if n <= 1:
        return new_c1, new_c2

    k = max(1, min(intensity, n - 1))
    removed = random.sample(tgt, k)
    tgt[:] = [v for v in tgt if v not in removed]

    for v in removed:
        pos = random.randrange(len(tgt) + 1)
        tgt.insert(pos, v)

    return new_c1, new_c2


def ils(coords, dist, time_limit, perturbation_intensity: int = 1):
    c1, c2 = generate_random_solution(coords)
    c1, c2 = local_search_solution(c1, c2, dist)
    best_sol, best_cost = (c1, c2), total_cost(c1, c2, dist)
    start = time.time()
    iters = 0
    while time.time() - start < time_limit:
        y_c1, y_c2 = perturbation_ils(c1, c2, intensity=perturbation_intensity)
        y_c1, y_c2 = local_search_solution(y_c1, y_c2, dist)
        y_cost = total_cost(y_c1, y_c2, dist)
        if y_cost < best_cost:
            c1, c2, best_cost = y_c1, y_c2, y_cost
        iters+=1

    return (c1, c2), best_cost, time.time() - start, iters

# ---------------- Large Neighbourhood Search -------------------------------

def perturbation_lns(c1, c2, removal_rate=0.3):
    def remove_random_vertices(cycle, k):
        removed = set(random.sample(cycle, k))
        cycle[:] = [v for v in cycle if v not in removed]
        return list(removed)

    new_c1, new_c2 = c1.copy(), c2.copy()
    r1 = max(1, int(len(new_c1) * removal_rate))
    r2 = max(1, int(len(new_c2) * removal_rate))
    removed_c1 = remove_random_vertices(new_c1, r1)
    removed_c2 = remove_random_vertices(new_c2, r2)
    return new_c1, removed_c1, new_c2, removed_c2

def lns(coords, dist, time_limit, removal_rate=0.3, is_local_also=False):
    c1, c2 = generate_random_solution(coords)
    c1, c2 = local_search_solution(c1, c2, dist)
    best_sol = (c1, c2)
    best_cost = total_cost(c1, c2, dist)
    start = time.time()
    iters=0
    while time.time() - start < time_limit:
        d_c1, rem_c1, d_c2, rem_c2 = perturbation_lns(c1, c2, removal_rate)
        weighted_regret_insertion_balanced(d_c1, d_c2, rem_c1 + rem_c2, dist, alpha=0.75)
        if is_local_also:
            d_c1, d_c2 = local_search_solution(d_c1, d_c2, dist)
        new_cost = total_cost(d_c1, d_c2, dist)
        if new_cost < best_cost:
            c1, c2, best_cost = d_c1, d_c2, new_cost
        iters+=1
    return (c1, c2), best_cost, time.time() - start, iters

def main():
    tsp_files = [
        "kro/kroA200.tsp",
        "kro/kroB200.tsp",
    ]

    runs = 10

    def process_instance(tsp_file):
        print(f"Przetwarzanie instancji: {tsp_file}")
        init_points = load_coords(tsp_file)
        distance_matrix = compute_distance_matrix(init_points)

        msls_results = []
        for i in range(runs):
            print(f"MSLS {i} dla {tsp_file}")
            sol, cost, t_time = msls(init_points, distance_matrix, num_iterations=200)
            msls_results.append((sol, cost, t_time))

        msls_times = [res[2] for res in msls_results]
        msls_costs = [res[1] for res in msls_results]
        best_msls_idx = msls_costs.index(min(msls_costs))
        best_msls_sol, best_msls_cost, _ = msls_results[best_msls_idx]
        avg_msls_time = sum(msls_times) / len(msls_times)


        ils_results = []
        for i in range(runs):
            print(f"ILS {i} dla {tsp_file}")
            sol, cost, t_time, iters = ils(init_points, distance_matrix, time_limit=avg_msls_time, perturbation_intensity=20)
            ils_results.append((sol, cost, t_time, iters))
            print(f"ILS count 1 {len(sol[0])} count 2 {len(sol[1])}")

        ils_costs = [res[1] for res in ils_results]
        ils_iterations = [res[3] for res in ils_results]
        best_ils_idx = ils_costs.index(min(ils_costs))
        best_ils_sol, best_ils_cost, _, _ = ils_results[best_ils_idx]
        avg_iters_ils = sum(ils_iterations) / len(ils_iterations)


        lnsa_results = []
        for i in range(runs):
            print(f"LNSa {i} dla {tsp_file}")
            sol, cost, t_time, iters = lns(init_points, distance_matrix, time_limit=avg_msls_time, removal_rate=0.6, is_local_also=False)
            lnsa_results.append((sol, cost, t_time, iters))
            #print(f"LNSa count 1 {len(sol[0])} count 2 {len(sol[1])}")

        lnsa_costs = [res[1] for res in lnsa_results]
        lnsa_iterations = [res[3] for res in lnsa_results]
        best_lnsa_idx = lnsa_costs.index(min(lnsa_costs))
        best_lnsa_sol, best_lnsa_cost, _, _ = lnsa_results[best_lnsa_idx]
        avg_iters_lnsa = sum(lnsa_iterations) / len(lnsa_iterations)


        lns_results = []
        for i in range(runs):
            print(f"LNS {i} dla {tsp_file}")
            sol, cost, t_time, iters = lns(init_points, distance_matrix, time_limit=avg_msls_time, removal_rate=0.6, is_local_also=True)
            lns_results.append((sol, cost, t_time, iters))
            #print(f"LNS count 1 {len(sol[0])} count 2 {len(sol[1])}")

        lns_costs = [res[1] for res in lns_results]
        lns_iterations = [res[3] for res in lns_results]
        best_lns_idx = lns_costs.index(min(lns_costs))
        best_lns_sol, best_lns_cost, _, _ = lns_results[best_lns_idx]
        avg_iters_lns = sum(lns_iterations) / len(lns_iterations)


        print(f"MSLS dla {tsp_file}:")
        print(f"  Min koszt: {min(msls_costs):.2f}")
        print(f"  Średni koszt: {sum(msls_costs) / len(msls_costs):.2f}")
        print(f"  Max koszt: {max(msls_costs):.2f}")
        print(f"  Średni czas: {avg_msls_time:.4f} s\n")
        print(f"ILS dla {tsp_file}:")
        print(f"  Min koszt: {min(ils_costs):.2f}")
        print(f"  Średni koszt: {sum(ils_costs)/len(ils_costs):.2f}")
        print(f"  Max koszt: {max(ils_costs):.2f}")
        print(f"  Średnia liczba perturbacji: {avg_iters_ils:.2f}\n")
        print(f"LNSa dla {tsp_file}:")
        print(f"  Min koszt: {min(lnsa_costs):.2f}")
        print(f"  Średni koszt: {sum(lnsa_costs)/len(lnsa_costs):.2f}")
        print(f"  Max koszt: {max(lnsa_costs):.2f}")
        print(f"  Średnia liczba perturbacji: {avg_iters_lnsa:.2f}\n")
        print(f"LNS dla {tsp_file}:")
        print(f"  Min koszt: {min(lns_costs):.2f}")
        print(f"  Średni koszt: {sum(lns_costs)/len(lns_costs):.2f}")
        print(f"  Max koszt: {max(lns_costs):.2f}")
        print(f"  Średnia liczba perturbacji: {avg_iters_lns:.2f}\n")


        plot_extended_solutions(
            tsp_file, init_points,
            best_msls_sol, best_msls_cost,
            best_ils_sol, best_ils_cost,
            best_lns_sol, best_lns_cost,
            best_lnsa_sol, best_lnsa_cost
        )

    for tsp_file in tsp_files:
        process_instance(tsp_file)

if __name__ == "__main__":
    main()
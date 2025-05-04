import os
import matplotlib.pyplot as plt


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

def plot_extended_solutions(tsp_file, coords, msls_sol, msls_cost, ils_sol, ils_cost, lns_sol, lns_cost, lnsa_sol, lnsa_cost):
    output_dir = "wykresy"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    basename = os.path.basename(tsp_file).replace('.tsp', '')

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_solution(ax, msls_sol[0], msls_sol[1], coords,
                  f"MSLS dla {basename}\nKoszt: {msls_cost:.2f}")
    plt.savefig(os.path.join(output_dir, f"{basename}_MSLS.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_solution(ax, ils_sol[0], ils_sol[1], coords,
                  f"ILS dla {basename}\nKoszt: {ils_cost:.2f}")
    plt.savefig(os.path.join(output_dir, f"{basename}_ILS.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_solution(ax, lns_sol[0], lns_sol[1], coords,
                  f"LNS dla {basename}\nKoszt: {lns_cost:.2f}")
    plt.savefig(os.path.join(output_dir, f"{basename}_LNS.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_solution(ax, lnsa_sol[0], lnsa_sol[1], coords,
                  f"LNSa dla {basename}\nKoszt: {lnsa_cost:.2f}")
    plt.savefig(os.path.join(output_dir, f"{basename}_LNSa.png"), dpi=300)
    plt.close()

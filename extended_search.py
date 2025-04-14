"""
extended_search.py

Implementacja trzech metod rozszerzonego lokalnego przeszukiwania:
  - MSLS (Multiple Start Local Search)
  - ILS (Iterated Local Search)
  - LNS (Large Neighborhood Search)

Przyjmujemy, że celem jest minimalizacja funkcji kosztu (np. suma odległości w cyklach).
W kodzie wykorzystujemy funkcje:
  - load_coords, generate_random_solution, generate_weighted_regret z modułu generate.py
  - total_cost, eucl_dist, local_steepest_edges_with_inter z modułu local_search.py
  - ewentualnie inne funkcje, do których dostęp mamy w wyniku poprzednich implementacji

Metody ILS i LNS przerywamy, gdy całkowity czas pracy osiągnie średni czas MSLS (dla tej instancji).
Każda z metod jest następnie testowana 10 razy (można tę liczbę regulować).
"""

import random
import time
import copy

# Importujemy funkcje z modułów, które już wcześniej stworzyliśmy
from construct import load_coords, generate_random_solution, generate_weighted_regret
from local_search import total_cost, eucl_dist, local_steepest_edges_with_inter


# =============================================================================
# Pomocnicze procedury
# =============================================================================
def local_search_solution(c1, c2, coords):
    """
    Wykonuje lokalne przeszukiwanie (najlepsze ze starych metod, np. LS z steepest edges)
    na rozwiązaniu reprezentowanym przez dwa cykle c1, c2.
    Zwraca (c1, c2) po poprawkach wykonanych przez lokalne przeszukiwanie.
    """
    new_c1, new_c2, _ = local_steepest_edges_with_inter(c1, c2, coords)
    return new_c1, new_c2


# -----------------------------------------------------------------------------
def perturbation_ils(c1, c2, intensity=1):
    """
    Perturbacja do metody ILS.

    Proponowana strategia:
      - W pewnej liczbie razy (intensity) wykonujemy losową modyfikację:
          * z 50% prawdopodobieństwem: wybieramy jeden z cykli i wykonujemy losową operację 2-opt (odwrócenie fragmentu)
          * lub, z 50%: wykonujemy losową wymianę jednego wierzchołka między cyklami

    Zwraca nowe, zaburzone rozwiązanie (c1, c2).
    """
    new_c1 = c1.copy()
    new_c2 = c2.copy()
    for _ in range(intensity):
        if random.random() < 0.5:
            # Losowy 2-opt – wybieramy cykl oraz dwa indeksy i odwracamy fragment
            if random.random() < 0.5 and len(new_c1) >= 4:
                cycle = new_c1
                idx = random.randint(0, len(cycle) - 1)
                jdx = random.randint(0, len(cycle) - 1)
                if idx == jdx:
                    continue
                i, j = min(idx, jdx), max(idx, jdx)
                # Wykonujemy odwrócenie fragmentu między i+1 a j (elementy w tej sekwencji)
                new_c1[i + 1:j + 1] = list(reversed(new_c1[i + 1:j + 1]))
            elif len(new_c2) >= 4:
                cycle = new_c2
                idx = random.randint(0, len(cycle) - 1)
                jdx = random.randint(0, len(cycle) - 1)
                if idx == jdx:
                    continue
                i, j = min(idx, jdx), max(idx, jdx)
                new_c2[i + 1:j + 1] = list(reversed(new_c2[i + 1:j + 1]))
        else:
            # Wymiana wierzchołków między cyklami (jeżeli oba nie są puste)
            if new_c1 and new_c2:
                i = random.randint(0, len(new_c1) - 1)
                j = random.randint(0, len(new_c2) - 1)
                new_c1[i], new_c2[j] = new_c2[j], new_c1[i]
    return new_c1, new_c2


# -----------------------------------------------------------------------------
def perturbation_lns(c1, c2, coords, removal_rate=0.3):
    """
    Perturbacja typu Destroy-Repair stosowana w LNS.

    Proces:
      1. "Destroy": losowo usuwamy około removal_rate (np. 30%) wierzchołków z każdego cyklu.
      2. "Repair": dla usuniętych wierzchołków, jeden po drugim, wyznaczamy najlepszą pozycję w którym
         dołożeniu (w cyklu 1 lub 2) wzrastanie kosztu (wyliczane na podstawie funkcji eucl_dist). Następnie wstawiamy
         wierzchołek w wybrane miejsce.

    Zwraca naprawione rozwiązanie (c1, c2).
    """
    new_c1 = c1.copy()
    new_c2 = c2.copy()
    # Obliczamy liczbę wierzchołków do usunięcia (minimum 1)
    num_remove_c1 = max(1, int(len(new_c1) * removal_rate))
    num_remove_c2 = max(1, int(len(new_c2) * removal_rate))

    # Losowo wybieramy indeksy do usunięcia (od największych do najmniejszych, aby usunięcie nie wpłynęło na pozycje)
    indices_remove_c1 = sorted(random.sample(range(len(new_c1)), num_remove_c1), reverse=True)
    indices_remove_c2 = sorted(random.sample(range(len(new_c2)), num_remove_c2), reverse=True)

    removed = []
    for idx in indices_remove_c1:
        removed.append(new_c1.pop(idx))
    for idx in indices_remove_c2:
        removed.append(new_c2.pop(idx))

    # Repair: dla każdego usuniętego wierzchołka szukamy najlepszego miejsca w obu cyklach (minimalny wzrost kosztu)
    for v in removed:
        best_increase = float('inf')
        best_cycle = None
        best_position = None
        # Sprawdzamy cykl 1
        if new_c1:
            for i in range(len(new_c1)):
                a = new_c1[i]
                b = new_c1[(i + 1) % len(new_c1)]
                increase = eucl_dist(coords[a], coords[v]) + eucl_dist(coords[v], coords[b]) - eucl_dist(coords[a],
                                                                                                         coords[b])
                if increase < best_increase:
                    best_increase = increase
                    best_cycle = 'c1'
                    best_position = i + 1
        else:
            best_increase = 0
            best_cycle = 'c1'
            best_position = 0

        # Sprawdzamy cykl 2
        if new_c2:
            for i in range(len(new_c2)):
                a = new_c2[i]
                b = new_c2[(i + 1) % len(new_c2)]
                increase = eucl_dist(coords[a], coords[v]) + eucl_dist(coords[v], coords[b]) - eucl_dist(coords[a],
                                                                                                         coords[b])
                if increase < best_increase:
                    best_increase = increase
                    best_cycle = 'c2'
                    best_position = i + 1
        else:
            best_increase = 0
            best_cycle = 'c2'
            best_position = 0

        if best_cycle == 'c1':
            new_c1.insert(best_position, v)
        else:
            new_c2.insert(best_position, v)
    return new_c1, new_c2


# =============================================================================
# Algorytmy rozszerzonego przeszukiwania
# =============================================================================
def msls(coords, num_iterations=200):
    """
    Multiple Start Local Search (MSLS)

    Powtarzamy:
      - generowanie losowego rozwiązania startowego (używamy generate_random_solution)
      - wykonanie lokalnego przeszukiwania (funkcja local_search_solution)
    w num_iterations (np. 200) iteracjach.

    Zwracamy:
      - najlepsze znalezione rozwiązanie (para cykli)
      - jego koszt (total_cost)
      - całkowity czas wykonania metody
    """
    best_sol = None
    best_cost = float('inf')
    start_time = time.time()

    for i in range(num_iterations):
        print(f"Iteration {i}")
        # Losowe rozwiązanie startowe
        c1, c2 = generate_random_solution(coords)
        # Ulepszamy rozwiązanie za pomocą LS
        ls_c1, ls_c2 = local_search_solution(c1, c2, coords)
        cost = total_cost(ls_c1, ls_c2, coords)
        if cost < best_cost:
            best_cost = cost
            best_sol = (ls_c1, ls_c2)

    total_time = time.time() - start_time
    return best_sol, best_cost, total_time


# -----------------------------------------------------------------------------
def ils(coords, time_limit, perturbation_intensity=1):
    """
    Iterated Local Search (ILS)

    1. Wygeneruj rozwiązanie początkowe x (losowe rozwiązanie i poprawienie przez LS).
    2. Powtarzaj (dopóki nie minie time_limit):
         a. y := kopia x
         b. Perturbacja – stosujemy funkcję perturbation_ils
         c. y := lokalne przeszukiwanie (LS) na y
         d. Jeżeli total_cost(y) < total_cost(x) (czyli rozwiązanie się poprawiło), to
               x := y
         e. Inkrementujemy licznik perturbacji.
    3. Zwracamy najlepsze rozwiązanie, koszt, czas pracy oraz liczbę wykonanych perturbacji.
    """
    # Rozwiązanie początkowe: losowe + LS
    c1, c2 = generate_random_solution(coords)
    c1, c2 = local_search_solution(c1, c2, coords)
    best_sol = (c1, c2)
    best_cost = total_cost(c1, c2, coords)
    iter_count = 0
    start_time = time.time()

    while time.time() - start_time < time_limit:
        # Tworzymy kopię obecnego rozwiązania i zaburzamy je
        y_c1, y_c2 = c1.copy(), c2.copy()
        y_c1, y_c2 = perturbation_ils(y_c1, y_c2, intensity=perturbation_intensity)
        # Ulepszamy rozwiązanie perturbowane – LS
        y_c1, y_c2 = local_search_solution(y_c1, y_c2, coords)
        y_cost = total_cost(y_c1, y_c2, coords)
        # Jeśli rozwiązanie uległo poprawie (czyli koszt jest mniejszy), to aktualizujemy
        if y_cost < best_cost:
            c1, c2 = y_c1, y_c2
            best_cost = y_cost
            best_sol = (c1, c2)
        iter_count += 1

    total_time = time.time() - start_time
    return best_sol, best_cost, total_time, iter_count


# -----------------------------------------------------------------------------
def lns(coords, time_limit, removal_rate=0.3):
    """
    Large Neighborhood Search (LNS)

    1. Wygeneruj rozwiązanie początkowe x (losowe rozwiązanie + LS; opcjonalnie LS)
    2. Powtarzaj (dopóki nie minie time_limit):
         a. y := kopia x
         b. Destroy – realizowane przez perturbation_lns (usunięcie ok. removal_rate wierzchołków z obu cykli)
         c. Repair – w ramach perturbation_lns (wstawienie usuniętych wierzchołków w najlepsze miejsca)
         d. Opcjonalnie – ulepszenie rozwiązania przez LS
         e. Jeśli total_cost(y) < total_cost(x) to x := y
         f. Inkrementacja licznika perturbacji.
    3. Zwracamy najlepsze znalezione rozwiązanie, koszt, czas pracy oraz liczbę perturbacji.
    """
    # Rozwiązanie początkowe
    c1, c2 = generate_random_solution(coords)
    c1, c2 = local_search_solution(c1, c2, coords)
    best_sol = (c1, c2)
    best_cost = total_cost(c1, c2, coords)
    iter_count = 0
    start_time = time.time()

    while time.time() - start_time < time_limit:
        y_c1, y_c2 = c1.copy(), c2.copy()
        # Destroy-Repair perturbation
        y_c1, y_c2 = perturbation_lns(y_c1, y_c2, coords, removal_rate=removal_rate)
        # Opcjonalnie: poprawiamy nowe rozwiązanie poprzez LS
        y_c1, y_c2 = local_search_solution(y_c1, y_c2, coords)
        y_cost = total_cost(y_c1, y_c2, coords)
        if y_cost < best_cost:
            c1, c2 = y_c1, y_c2
            best_cost = y_cost
            best_sol = (c1, c2)
        iter_count += 1

    total_time = time.time() - start_time
    return best_sol, best_cost, total_time, iter_count


# =============================================================================
# Funkcja main – przykładowe wykonanie eksperymentów
# =============================================================================
def main():
    # Definiujemy instancje – zgodnie z poleceniem używamy np. kroA200 i kroB200
    tsp_files = [
        "kro/kroA200.tsp",
        "kro/krob200.tsp"
    ]

    runs = 3  # każda metoda wykonywana 10 razy
    print("=== Eksperyment rozszerzonego przeszukiwania ===\n")

    for tsp_file in tsp_files:
        print(f"Przetwarzam instancję: {tsp_file}")
        coords = load_coords(tsp_file)

        # ---------------------------------------------------------
        # MSLS: wykonujemy 200 iteracji LS dla każdego losowego startu
        msls_times = []
        msls_costs = []
        best_msls_sol = None
        best_msls_cost = float('inf')
        for i in range(runs):
            print(f"Run no. {i}")
            sol, cost, t_time = msls(coords, num_iterations=20)
            msls_times.append(t_time)
            msls_costs.append(cost)
            if cost < best_msls_cost:
                best_msls_cost = cost
                best_msls_sol = sol
        avg_time = sum(msls_times) / len(msls_times)
        print(f"MSLS: Średni czas = {avg_time:.4f} s, najlepszy koszt = {best_msls_cost:.2f}")

        # ---------------------------------------------------------
        # ILS: warunkiem stopu jest osiągnięcie czasu równego średniemu czasowi MSLS
        ils_times = []
        ils_costs = []
        ils_iterations = []
        best_ils_sol = None
        best_ils_cost = float('inf')
        for i in range(runs):
            sol, cost, t_time, iters = ils(coords, time_limit=avg_time, perturbation_intensity=1)
            ils_times.append(t_time)
            ils_costs.append(cost)
            ils_iterations.append(iters)
            if cost < best_ils_cost:
                best_ils_cost = cost
                best_ils_sol = sol
        avg_iters_ils = sum(ils_iterations) / len(ils_iterations)
        print(f"ILS: Najlepszy koszt = {best_ils_cost:.2f}, średnia liczba perturbacji = {avg_iters_ils:.2f}")

        # ---------------------------------------------------------
        # LNS: warunkiem stopu jest osiągnięcie czasu równego średniemu czasowi MSLS
        lns_times = []
        lns_costs = []
        lns_iterations = []
        best_lns_sol = None
        best_lns_cost = float('inf')
        for i in range(runs):
            sol, cost, t_time, iters = lns(coords, time_limit=avg_time, removal_rate=0.3)
            lns_times.append(t_time)
            lns_costs.append(cost)
            lns_iterations.append(iters)
            if cost < best_lns_cost:
                best_lns_cost = cost
                best_lns_sol = sol
        avg_iters_lns = sum(lns_iterations) / len(lns_iterations)
        print(f"LNS: Najlepszy koszt = {best_lns_cost:.2f}, średnia liczba perturbacji = {avg_iters_lns:.2f}\n")


if __name__ == "__main__":
    main()

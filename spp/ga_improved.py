"""Improved binary GA for the Set Partitioning Problem.

Three things change compared with the standard GA:

1. Initialisation - individuals are built with a randomised greedy heuristic
   (cheapest cost per newly covered leg, never covering a leg twice), so the
   population starts close to the feasible region instead of at random.
2. Constraint handling - stochastic ranking (Runarsson & Yao, 2000) replaces
   the static penalty, so there is no penalty weight to tune.
3. A light local clean-up removes redundant rotations from some children.
"""

import random

import numpy as np


def evaluate(pop, cost, matrix):
    pop_i = pop.astype(np.int64)
    obj = pop_i @ cost.astype(np.int64)
    coverage = pop_i @ matrix.T.astype(np.int64)
    violations = np.abs(coverage - 1).sum(axis=1)
    return obj, violations


def stochastic_rank(obj, viol, pf):
    """Bubble-sort style stochastic ranking.

    Two neighbours are compared on cost if both are feasible (or with
    probability pf), otherwise on constraint violation.
    """
    n = len(obj)
    order = list(range(n))

    for _ in range(n):
        swapped = False
        for i in range(n - 1):
            a, b = order[i], order[i + 1]
            if (viol[a] == 0 and viol[b] == 0) or random.random() < pf:
                worse = obj[a] > obj[b] or (obj[a] == obj[b] and viol[a] > viol[b])
            else:
                worse = viol[a] > viol[b] or (viol[a] == viol[b] and obj[a] > obj[b])
            if worse:
                order[i], order[i + 1] = b, a
                swapped = True
        if not swapped:
            break

    return np.array(order)


def build_maps(matrix):
    rows, cols = matrix.shape
    row_to_cols = [np.where(matrix[r] == 1)[0].tolist() for r in range(rows)]
    col_to_rows = [np.where(matrix[:, c] == 1)[0] for c in range(cols)]
    return row_to_cols, col_to_rows


def greedy_start(row_map, col_map, cost, rows, cols):
    genome = np.zeros(cols, dtype=bool)
    uncovered = set(range(rows))
    covered = set()

    while uncovered:
        r = random.choice(tuple(uncovered))
        candidates = [c for c in row_map[r]
                      if not any(x in covered for x in col_map[c])]
        if not candidates:
            return None

        scores = []
        for c in candidates:
            gain = max(sum(1 for r2 in col_map[c] if r2 in uncovered), 1)
            scores.append(cost[c] / gain + random.random() * 0.2)

        best = candidates[int(np.argmin(scores))]
        genome[best] = True
        for rr in col_map[best]:
            covered.add(int(rr))
            uncovered.discard(int(rr))

    return genome


def drop_redundant(genome, col_map, rows):
    """Remove rotations whose legs are all covered by something else anyway."""
    selected = set(np.where(genome)[0])
    cover = np.zeros(rows, dtype=int)
    for c in selected:
        cover[col_map[c]] += 1

    for c in list(selected):
        if np.all(cover[col_map[c]] > 1):
            selected.remove(c)
            cover[col_map[c]] -= 1

    if np.all(cover == 1):
        out = np.zeros_like(genome)
        out[list(selected)] = True
        return out
    return genome


def tournament_select(pop, ranks, k):
    n = len(pop)
    chosen = np.empty(n, dtype=int)
    for i in range(n):
        entrants = np.random.choice(n, k, replace=False)
        chosen[i] = entrants[np.argmin(ranks[entrants])]
    return pop[chosen]


def crossover(parents):
    n, cols = parents.shape
    children = np.empty_like(parents)
    for i in range(0, n, 2):
        p1, p2 = parents[i], parents[(i + 1) % n]
        mask = np.random.rand(cols) < 0.5
        children[i] = np.where(mask, p1, p2)
        if i + 1 < n:
            children[i + 1] = np.where(mask, p2, p1)
    return children


def mutate(children, pm):
    return np.logical_xor(children, np.random.rand(*children.shape) < pm)


def run_ga(cost, matrix, pop_size=50, generations=200, tournament_k=3,
           pm=None, pf=0.45, local_rate=0.3, verbose=False):
    rows, cols = matrix.shape
    if not pm or pm <= 0:
        pm = 1.0 / cols

    row_map, col_map = build_maps(matrix)

    pop = np.zeros((pop_size, cols), dtype=bool)
    for i in range(pop_size):
        for _ in range(50):
            g = greedy_start(row_map, col_map, cost, rows, cols)
            if g is not None:
                pop[i] = g
                break
        else:
            pop[i] = np.random.rand(cols) < 0.1   # greedy kept failing

    best_cost, best_genome = None, None

    for gen in range(generations):
        obj, viol = evaluate(pop, cost, matrix)

        feasible = np.where(viol == 0)[0]
        if feasible.size:
            idx = feasible[np.argmin(obj[feasible])]
            if best_cost is None or obj[idx] < best_cost:
                best_cost, best_genome = int(obj[idx]), pop[idx].copy()

        order = stochastic_rank(obj, viol, pf)
        rank_pos = np.empty(pop_size, dtype=int)
        rank_pos[order] = np.arange(pop_size)

        children = mutate(crossover(tournament_select(pop, rank_pos, tournament_k)), pm)
        for i in range(pop_size):
            if random.random() < local_rate:
                children[i] = drop_redundant(children[i], col_map, rows)

        merged = np.vstack([pop, children])
        m_obj, m_viol = evaluate(merged, cost, matrix)
        pop = merged[stochastic_rank(m_obj, m_viol, pf)[:pop_size]]

        if verbose and (gen + 1) % 20 == 0:
            print(f"  gen {gen + 1:3d} | best feasible cost {best_cost}")

    if best_genome is None:
        return None, None
    return np.where(best_genome)[0].tolist(), best_cost


def improved_genetic_algorithm(rows, columns, matrix, cost, **kwargs):
    return run_ga(cost, matrix, **kwargs)

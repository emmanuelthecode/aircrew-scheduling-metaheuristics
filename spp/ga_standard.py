"""Standard binary GA with a static penalty function.

This is the "textbook" baseline: random initial population, penalised
fitness, tournament selection, uniform crossover, bit-flip mutation and
(mu + lambda) replacement. The penalty charges 1x for every extra time a leg
is covered and 5x for every leg left uncovered.

My first version stored each individual as a (bits, cost) tuple in an object
array, which made a single run on NW41 take over a minute. The logic below is
the same, it just keeps the whole population in one 2D boolean array so numpy
can do the work.
"""

import numpy as np


def penalised_cost(pop, matrix, cost, penalty):
    pop_i = pop.astype(np.int64)
    raw = pop_i @ cost
    coverage = pop_i @ matrix.T
    over = np.maximum(coverage - 1, 0).sum(axis=1)
    under = (coverage == 0).sum(axis=1)
    return raw + (over + 5 * under) * penalty


def tournament_selection(pop, fitness, k):
    n = len(pop)
    entrants = np.random.randint(0, n, size=(n, k))
    winners = entrants[np.arange(n), np.argmin(fitness[entrants], axis=1)]
    return pop[winners]


def uniform_crossover(parents, generation):
    # bias the mask a bit more towards the first parent as the run goes on
    if generation < 40:
        rate = 0.5
    elif generation < 70:
        rate = 0.6
    else:
        rate = 0.7

    n, cols = parents.shape
    pairs = (n + 1) // 2
    # two different random parents per pair
    a = np.random.randint(0, n, size=pairs)
    b = (a + np.random.randint(1, n, size=pairs)) % n
    mask = np.random.rand(pairs, cols) < rate

    first = np.where(mask, parents[a], parents[b])
    second = np.where(mask, parents[b], parents[a])
    children = np.empty((pairs * 2, cols), dtype=parents.dtype)
    children[0::2], children[1::2] = first, second
    return children[:n]


def binary_genetic_algorithm(rows, columns, matrix, cost,
                             pop_size=2000, penalty=1000, generations=200,
                             tournament_k=2, mutation_rate=None, verbose=False):
    if mutation_rate is None:
        mutation_rate = 1 / columns

    pop = np.random.rand(pop_size, columns) < 0.5
    fitness = penalised_cost(pop, matrix, cost, penalty)

    for gen in range(generations):
        parents = tournament_selection(pop, fitness, tournament_k)
        children = uniform_crossover(parents, gen)
        children ^= np.random.rand(*children.shape) < mutation_rate

        child_fitness = penalised_cost(children, matrix, cost, penalty)

        # (mu + lambda): keep the best pop_size out of parents + children
        merged = np.vstack([pop, children])
        merged_fit = np.concatenate([fitness, child_fitness])
        keep = np.argsort(merged_fit, kind="stable")[:pop_size]
        pop, fitness = merged[keep], merged_fit[keep]

        if verbose and (gen + 1) % 10 == 0:
            print(f"  gen {gen + 1:3d} | best penalised cost {fitness[0]}")

    best = pop[np.argmin(fitness)]
    return np.where(best)[0].tolist(), int(fitness.min())

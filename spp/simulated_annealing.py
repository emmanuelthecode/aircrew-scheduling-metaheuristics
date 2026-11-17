"""Simulated annealing for the Set Partitioning Problem.

The search only ever moves between feasible schedules. A neighbour is made by
dropping one or two rotations from the current schedule and then greedily
re-filling the uncovered legs with random rotations that don't clash with
anything already chosen. If the repair gets stuck we just start again from a
fresh random schedule.
"""

import math
import random

import numpy as np


def candidate_solution(matrix, cost, rows, columns, max_restarts=10_000):
    """Build a random feasible schedule (column indices) and its cost."""
    for _ in range(max_restarts):
        chosen = np.zeros(columns, dtype=bool)
        covered = np.zeros(rows, dtype=bool)

        while not covered.all():
            # columns that don't touch any leg we've already covered
            free = np.where(np.all(matrix[covered, :] == 0, axis=0))[0]
            if free.size == 0:
                break
            chosen[np.random.choice(free)] = True
            covered = (matrix @ chosen) > 0

        if covered.all():
            sol = np.where(chosen)[0]
            return sol, int(cost[sol].sum())

    raise RuntimeError("couldn't build a feasible starting schedule")


def neighbour_selection(solution, matrix, cost, rows, columns):
    chosen = np.zeros(columns, dtype=bool)
    chosen[solution] = True

    # drop 1-2 rotations
    selected = np.where(chosen)[0]
    if selected.size:
        k = min(random.randint(1, 2), selected.size)
        chosen[np.random.choice(selected, k, replace=False)] = False

    covered = (matrix @ chosen) > 0

    # repair
    while not covered.all():
        free = np.where(np.all(matrix[covered, :] == 0, axis=0))[0]
        if free.size == 0:
            return candidate_solution(matrix, cost, rows, columns)
        chosen[np.random.choice(free)] = True
        covered = (matrix @ chosen) > 0

    sol = np.where(chosen)[0]
    return sol, int(cost[sol].sum())


def simulated_annealing(rows, columns, matrix, cost,
                        temp=10_000, cooling=0.995, iterations=10_000):
    current, current_cost = candidate_solution(matrix, cost, rows, columns)
    best, best_cost = current, current_cost

    for _ in range(iterations):
        neighbour, neighbour_cost = neighbour_selection(current, matrix, cost, rows, columns)

        diff = current_cost - neighbour_cost   # > 0 means the neighbour is cheaper
        if diff > 0 or math.exp(diff / temp) > random.random():
            current, current_cost = neighbour, neighbour_cost
            if current_cost < best_cost:
                best, best_cost = current, current_cost

        temp *= cooling

    return best.tolist(), best_cost

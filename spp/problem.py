"""Loading and evaluating Set Partitioning Problem instances.

The OR-Library files (Beasley, 1990) look like this:

    <rows> <columns>
    <cost> <k> <row_1> ... <row_k>     <- one line per column
    ...

Rows are flight legs, columns are candidate crew rotations. A feasible
schedule picks a set of columns so that every row is covered exactly once.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    rows: int
    columns: int
    cost: np.ndarray      # shape (columns,)
    matrix: np.ndarray    # shape (rows, columns), 0/1 incidence matrix
    name: str = ""


def load_instance(path):
    with open(path) as f:
        data = list(map(int, f.read().split()))

    m, n = data[0], data[1]
    cost = np.zeros(n, dtype=int)
    matrix = np.zeros((m, n), dtype=int)

    idx = 2
    for j in range(n):
        cost[j] = data[idx]
        k = data[idx + 1]
        covered = data[idx + 2: idx + 2 + k]
        idx += 2 + k
        for r in covered:
            matrix[r - 1, j] = 1

    return Instance(m, n, cost, matrix, name=str(path))


def parser(path):
    """Old tuple-style interface: returns (rows, columns, cost, matrix)."""
    inst = load_instance(path)
    return inst.rows, inst.columns, inst.cost, inst.matrix


def to_bits(solution, columns):
    """Accepts either a list of column indices or a boolean vector."""
    solution = np.asarray(solution)
    if solution.dtype == bool and solution.size == columns:
        return solution
    bits = np.zeros(columns, dtype=bool)
    if solution.size:
        bits[solution.astype(int)] = True
    return bits


def solution_cost(solution, cost):
    bits = to_bits(solution, len(cost))
    return int(cost[bits].sum())


def row_coverage(solution, matrix):
    bits = to_bits(solution, matrix.shape[1])
    return matrix @ bits.astype(int)


def is_feasible(solution, matrix):
    return bool(np.all(row_coverage(solution, matrix) == 1))

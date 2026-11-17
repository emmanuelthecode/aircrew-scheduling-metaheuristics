import os
import random

import numpy as np
import pytest

from spp import ALGORITHMS, load_instance, is_feasible, solution_cost

# 4 legs, 5 rotations. Optimal schedule is columns {0, 3} with cost 4.
TOY = """4 5
3 2 1 2
4 2 1 2
5 1 3
1 2 3 4
9 4 1 2 3 4
"""


@pytest.fixture
def toy(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_text(TOY)
    return load_instance(p)


def test_parser(toy):
    assert (toy.rows, toy.columns) == (4, 5)
    assert toy.cost.tolist() == [3, 4, 5, 1, 9]
    assert toy.matrix[:, 4].tolist() == [1, 1, 1, 1]
    assert toy.matrix[:, 2].tolist() == [0, 0, 1, 0]


def test_feasibility(toy):
    assert is_feasible([0, 3], toy.matrix)
    assert is_feasible([4], toy.matrix)
    assert not is_feasible([0, 1, 3], toy.matrix)   # legs 1-2 covered twice
    assert not is_feasible([0], toy.matrix)         # legs 3-4 uncovered
    assert solution_cost([0, 3], toy.cost) == 4


@pytest.mark.parametrize("key", ["sa", "bga", "ibga"])
def test_solvers_find_optimum_on_toy(toy, key):
    random.seed(1)
    np.random.seed(1)
    _, solve = ALGORITHMS[key]
    kwargs = {"iterations": 300} if key == "sa" else {"generations": 30}
    if key == "bga":
        kwargs["pop_size"] = 60
    sol, _ = solve(toy.rows, toy.columns, toy.matrix, toy.cost, **kwargs)
    assert is_feasible(sol, toy.matrix)
    assert solution_cost(sol, toy.cost) == 4


@pytest.mark.skipif(not os.path.exists("data/sppnw41.txt"), reason="run from repo root")
def test_real_instance_loads():
    inst = load_instance("data/sppnw41.txt")
    assert (inst.rows, inst.columns) == (17, 197)
    assert inst.matrix.sum(axis=0).min() >= 1

# Airline Crew Scheduling with Metaheuristics

[![tests](https://github.com/emmanuelthecode/aircrew-scheduling-metaheuristics/actions/workflows/tests.yml/badge.svg)](https://github.com/emmanuelthecode/aircrew-scheduling-metaheuristics/actions/workflows/tests.yml) ![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Airlines have to assign crews to every flight leg they operate, and crews are expensive. Given a big list of possible crew rotations (each one covers some legs and has a cost), the job is to pick a set of rotations so that **every leg is covered exactly once** at the **lowest total cost**. That's the Set Partitioning Problem, and it's NP-hard.

This repo has the three solvers I built and compared for my Evolutionary Computation module at the University of Birmingham:

| Key    | Algorithm | Constraint handling |
|--------|-----------|---------------------|
| `sa`   | Simulated Annealing | Only ever visits feasible schedules (drop 1–2 rotations, then repair) |
| `bga`  | Standard Binary Genetic Algorithm | Static penalty function |
| `ibga` | Improved Binary GA | Greedy initialisation + stochastic ranking (Runarsson & Yao, 2000) + redundant-column clean-up |

They're tested on three real airline instances from Beasley's OR-Library: **NW41**, **NW42** and **NW43**.

The full write-up (pseudocode, flowcharts, parameter choices and a discussion of ranking replacement vs. stochastic ranking) is in [`report/crew-scheduling-report.pdf`](report/crew-scheduling-report.pdf).

## Formulation

```
minimise    Σ c_j · x_j
subject to  A·x = 1          (every leg covered exactly once)
            x_j ∈ {0, 1}
```

`A` is the leg × rotation incidence matrix and `c_j` is the cost of rotation `j`.

## Results (30 runs each)

| Instance | Known optimum | SA best / mean ± sd | BGA best / mean ± sd | IBGA best / mean ± sd |
|----------|---------------|---------------------|----------------------|------------------------|
| NW41 | 11307 | **11307** / 11335.7 ± 52.0 | 11307 / 11441.4 ± 256.3 | 11307 / 11622.1 ± 561.9 |
| NW42 | 7656  | **7656** / 7686.1 ± 48.7   | 7656 / 8588.6 ± 709.9   | 7656 / 7890.4 ± 337.3  |
| NW43 | 8904  | **8904** / 9110.9 ± 201.8  | 8904 / 9995.5 ± 519.0   | 8904 / 9378.3 ± 287.2  |

What I took away from it:

- All three can find the optimum, so the real difference is **reliability**, not best-case quality.
- **Simulated annealing was the most consistent** by a long way. Staying inside the feasible region and repairing after each move works really well on these instances.
- The **standard GA** was the least reliable. The penalty weight is a guess, and a random initial population spends most of its time just trying to become feasible.
- The **improved GA** fixed a lot of that on NW42 and NW43 (lower mean, roughly half the spread), but on NW41 it was worse, so the improvements are clearly problem-dependent.

## Getting started

```bash
git clone https://github.com/emmanuelthecode/aircrew-scheduling-metaheuristics.git
cd aircrew-scheduling-metaheuristics
pip install -r requirements.txt
```

Run a solver:

```bash
python run.py sa sppnw41.txt              # 30 runs (default)
python run.py ibga sppnw43.txt --runs 10
python run.py bga sppnw42.txt --runs 5 --seed 1 --verbose
```

Instance names are looked up in `data/` if you don't give a full path. Each run uses seed `base + i`, so results are reproducible.

Example output:

```
Simulated Annealing on sppnw41.txt (17 legs, 197 rotations), 5 runs

run 01 | cost  11307 |  5 rotations | 0.62s
run 02 | cost  11307 |  5 rotations | 0.63s
...
summary
  feasible runs : 5/5
  best          : 11307
  worst         : 11430
  mean          : 11331.6
  std dev       : 49.2
```

Tests:

```bash
python -m pytest
```

## Project layout

```
spp/
  problem.py               instance loader, cost and feasibility checks
  simulated_annealing.py   SA with drop-and-repair neighbourhood
  ga_standard.py           penalty-based binary GA
  ga_improved.py           greedy init + stochastic ranking GA
data/                      OR-Library instances NW41, NW42, NW43
report/                    the coursework report (PDF)
tests/                     pytest suite with a hand-checked toy instance
run.py                     command-line benchmark runner
```

## Notes since the original submission

The algorithms and parameters are the same as in the report, but I tidied the code afterwards:

- The three copies of the parser are now one module (`spp/problem.py`).
- The standard GA kept the population as a NumPy object array of `(bits, cost)` tuples. It now uses a single 2D boolean array, which took one NW41 run from about **75s down to about 1.5s** without changing the algorithm.
- SA's start-solution builder used recursion to retry; it's a loop now, so it can't hit Python's recursion limit.
- `run.py` checks every returned schedule for feasibility instead of trusting the penalised cost.

Because of the refactor, timings won't match Table 3 in the report.

## References

- J. E. Beasley (1990). OR-Library: distributing test problems by electronic mail. *Journal of the Operational Research Society*, 41(11).
- P. C. Chu and J. E. Beasley (1998). Constraint handling in genetic algorithms: the set partitioning problem. *Journal of Heuristics*, 4.
- T. P. Runarsson and X. Yao (2000). Stochastic ranking for constrained evolutionary optimization. *IEEE Transactions on Evolutionary Computation*, 4(3).
- S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi (1983). Optimization by simulated annealing. *Science*, 220(4598).

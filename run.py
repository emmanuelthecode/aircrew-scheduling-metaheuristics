#!/usr/bin/env python3
"""Run one of the solvers several times on an instance and print summary stats.

    python run.py sa sppnw41.txt
    python run.py ibga data/sppnw43.txt --runs 10 --seed 7
"""

import argparse
import os
import random
import time

import numpy as np

from spp import ALGORITHMS, load_instance, is_feasible, solution_cost

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    if os.path.exists(path):
        return path
    candidate = os.path.join(HERE, "data", path)
    if os.path.exists(candidate):
        return candidate
    raise SystemExit(f"can't find '{path}' (looked in the current folder and data/)")


def benchmark(key, path, runs, seed, verbose=False):
    name, solve = ALGORITHMS[key]
    inst = load_instance(path)
    print(f"\n{name} on {os.path.basename(path)} "
          f"({inst.rows} legs, {inst.columns} rotations), {runs} runs\n")

    costs, times = [], []
    for run in range(runs):
        random.seed(seed + run)
        np.random.seed(seed + run)

        start = time.perf_counter()
        kwargs = {"verbose": True} if verbose and key != "sa" else {}
        solution, _ = solve(inst.rows, inst.columns, inst.matrix, inst.cost, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if solution is None or not is_feasible(solution, inst.matrix):
            print(f"run {run + 1:02d} | no feasible schedule found ({elapsed:.2f}s)")
            continue

        cost = solution_cost(solution, inst.cost)
        costs.append(cost)
        print(f"run {run + 1:02d} | cost {cost:6d} | {len(solution):2d} rotations | {elapsed:.2f}s")

    print("\nsummary")
    print(f"  feasible runs : {len(costs)}/{runs}")
    if costs:
        print(f"  best          : {min(costs)}")
        print(f"  worst         : {max(costs)}")
        print(f"  mean          : {np.mean(costs):.1f}")
        print(f"  std dev       : {np.std(costs):.1f}")
    print(f"  time          : {sum(times):.1f}s total, {np.mean(times):.2f}s per run")


def main():
    ap = argparse.ArgumentParser(description="Benchmark SPP metaheuristics on OR-Library crew scheduling instances")
    ap.add_argument("algorithm", choices=ALGORITHMS.keys(), help="sa, bga or ibga")
    ap.add_argument("instance", help="instance file, e.g. sppnw41.txt (looked up in data/ if not found)")
    ap.add_argument("--runs", type=int, default=30, help="independent runs (default 30)")
    ap.add_argument("--seed", type=int, default=2026, help="base seed; run i uses seed + i")
    ap.add_argument("--verbose", action="store_true", help="print GA progress")
    args = ap.parse_args()

    benchmark(args.algorithm, resolve(args.instance), args.runs, args.seed, args.verbose)


if __name__ == "__main__":
    main()

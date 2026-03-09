#!/usr/bin/env python3
"""Simulate an IV dataset and estimate ATE bounds with subsampling CIs."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure local source tree is imported when run as:
# `python iv_ate_ci_subsampling.py` from the `examples/` directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autobounds import DAG, causalProblem


def simulate_iv_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Binary latent confounder and binary instrument.
    u = rng.binomial(1, 0.5, n)
    z = rng.binomial(1, 0.5, n)

    # Treatment depends on IV and confounder.
    p_x = 0.1 + 0.45 * z + 0.25 * u
    p_x = np.clip(p_x, 0.01, 0.99)
    x = rng.binomial(1, p_x)

    # Outcome depends on treatment and confounder.
    p_y = 0.1 + 0.35 * x + 0.30 * u
    p_y = np.clip(p_y, 0.01, 0.99)
    y = rng.binomial(1, p_y)

    return pd.DataFrame({"Z": z, "X": x, "Y": y})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500, help="Number of simulated rows.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--nsamples", type=int, default=80, help="Subsampling CI replications.")
    parser.add_argument("--ci-workers", type=int, default=16, help="Parallel workers inside CI solve.")
    parser.add_argument(
        "--subsample-rate",
        type=float,
        default=0.7,
        help="Exponent for default subsample size m=floor(n**rate).",
    )
    parser.add_argument(
        "--maxtime",
        type=float,
        default=15.0,
        help="Max solver seconds per optimization run.",
    )
    args = parser.parse_args()

    df = simulate_iv_data(args.n, args.seed)

    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob="U")

    problem = causalProblem(dag)
    problem.set_ate("X", "Y")
    problem.read_data(raw=df)

    result = problem.solve(
        ci=True,
        nsamples=args.nsamples,
        ci_workers=args.ci_workers,
        subsample_rate=args.subsample_rate,
        ci_method="recentered_subsampling",
        maxtime=args.maxtime,
        verbose_optimizer=False,
        verbose_result=False,
        limits=[-1, 1],
    )

    print("Point bounds (dual):", result["point lb dual"], result["point ub dual"])
    print("Point bounds (primal):", result["point lb primal"], result["point ub primal"])
    print("95% CI:", result["2.5% lb bounds"], result["97.5% ub bounds"])
    print("98% CI:", result["1% lb bounds"], result["99% ub bounds"])


if __name__ == "__main__":
    main()

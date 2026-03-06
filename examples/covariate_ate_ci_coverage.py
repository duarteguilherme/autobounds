#!/usr/bin/env python3
"""Coverage experiment for ATE bounds with a covariate X."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autobounds import DAG, causalProblem


def simulate_covariate_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    x = rng.binomial(1, 0.5, n)
    u = rng.binomial(1, 0.5, n)
    z = rng.binomial(1, 0.5, n)

    p_d = 0.1 + 0.35 * z + 0.20 * x + 0.20 * u
    p_d = np.clip(p_d, 0.01, 0.99)
    d = rng.binomial(1, p_d)

    p_y = 0.05 + 0.30 * d + 0.20 * x + 0.25 * u
    p_y = np.clip(p_y, 0.01, 0.99)
    y = rng.binomial(1, p_y)

    return pd.DataFrame({"X": x, "Z": z, "D": d, "Y": y})


def build_problem(df: pd.DataFrame) -> causalProblem:
    dag = DAG()
    dag.from_structure("X -> D, X -> Y, Z -> D, D -> Y, U -> D, U -> Y", unob="U")
    problem = causalProblem(dag)
    problem.set_ate("D", "Y")
    problem.read_data(raw=df, covariates=["X"], inference=True)
    return problem


def solve_point(problem: causalProblem, maxtime: float) -> dict:
    # CI with covariates is intentionally delegated to Bounder for now.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return problem.get_bounder("default").solve(
            ci=False,
            maxtime=maxtime,
            verbose_optimizer=False,
            verbose_result=False,
            limits=[-1, 1],
        )


def solve_ci(problem: causalProblem, nsamples: int, maxtime: float) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return problem.get_bounder("default").solve(
            ci=True,
            nsamples=nsamples,
            maxtime=maxtime,
            verbose_optimizer=False,
            verbose_result=False,
            limits=[-1, 1],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=75, help="Number of Monte Carlo datasets.")
    parser.add_argument("--n", type=int, default=600, help="Rows per Monte Carlo dataset.")
    parser.add_argument("--nsamples", type=int, default=100, help="CI inner reps.")
    parser.add_argument("--true-n", type=int, default=120000, help="Rows for true-bound proxy.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rep-seed-base", type=int, default=7000)
    parser.add_argument("--maxtime", type=float, default=6.0)
    parser.add_argument("--print-cis", action="store_true")
    args = parser.parse_args()

    print(f"Estimating true bounds with true_n={args.true_n} ...", flush=True)
    true_df = simulate_covariate_data(args.true_n, args.seed)
    true_out = solve_point(build_problem(true_df), maxtime=max(args.maxtime, 12.0))
    true_lb = float(true_out["point lb dual"])
    true_ub = float(true_out["point ub dual"])
    print(f"true_lb={true_lb:.8f} true_ub={true_ub:.8f}", flush=True)

    cover_lb = cover_ub = cover_joint = 0
    rep_rows = []
    for r in range(args.r):
        df = simulate_covariate_data(args.n, args.rep_seed_base + r)
        out = solve_ci(build_problem(df), nsamples=args.nsamples, maxtime=args.maxtime)
        lb = float(out["2.5% lb bounds"])
        ub = float(out["97.5% ub bounds"])
        lb_ok = lb <= true_lb
        ub_ok = ub >= true_ub
        both = lb_ok and ub_ok
        cover_lb += int(lb_ok)
        cover_ub += int(ub_ok)
        cover_joint += int(both)
        rep_rows.append((r, lb, ub, both))
        if (r + 1) % 10 == 0 or (r + 1) == args.r:
            print(
                f"progress {r+1}/{args.r} "
                f"lb={cover_lb/(r+1):.4f} ub={cover_ub/(r+1):.4f} joint={cover_joint/(r+1):.4f}",
                flush=True,
            )

    if args.print_cis:
        print("----- PER-REPLICATE CIs -----")
        for rid, lb, ub, both in rep_rows:
            print(
                f"rep={rid:04d} "
                f"CI=[{lb:.6f}, {ub:.6f}] "
                f"true=[{true_lb:.6f}, {true_ub:.6f}] "
                f"cover={both}"
            )

    print("----- FINAL -----")
    print(f"R={args.r} n={args.n} nsamples={args.nsamples}")
    print(f"lb_coverage={cover_lb/args.r:.6f}")
    print(f"ub_coverage={cover_ub/args.r:.6f}")
    print(f"joint_coverage={cover_joint/args.r:.6f}")


if __name__ == "__main__":
    main()

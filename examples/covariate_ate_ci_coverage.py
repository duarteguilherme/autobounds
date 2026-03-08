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


def build_problem_x(df: pd.DataFrame) -> causalProblem:
    dag = DAG()
    dag.from_structure("Z -> D, D -> Y, U -> D, U -> Y", unob="U")
    problem = causalProblem(dag)
    problem.set_ate("D", "Y")
    problem.read_data(raw=df[["Z", "D", "Y"]], inference=True)
    return problem


def solve_point(problem: causalProblem, maxtime: float) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return problem.solve(
            ci=False,
            maxtime=maxtime,
            verbose_optimizer=False,
            verbose_result=False,
            limits=[-1, 1],
        )


def solve_ci(
    problem: causalProblem,
    nsamples: int,
    ci_workers: int,
    subsample_rate: float,
    maxtime: float,
) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return problem.solve(
            ci=True,
            nsamples=nsamples,
            ci_workers=ci_workers,
            subsample_rate=subsample_rate,
            ci_method="recentered_subsampling",
            maxtime=maxtime,
            verbose_optimizer=False,
            verbose_result=False,
            limits=[-1, 1],
        )


def _safe_ratio(num, den):
    return float(num) / float(den) if den > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=75, help="Number of Monte Carlo datasets.")
    parser.add_argument("--n", type=int, default=600, help="Rows per Monte Carlo dataset.")
    parser.add_argument("--b", type=int, default=500, help="Subsampling reps per dataset.")
    parser.add_argument("--nsamples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ci-workers", type=int, default=16, help="Parallel workers inside each CI solve.")
    parser.add_argument("--subsample-rate", type=float, default=(2.0 / 3.0))
    parser.add_argument("--true-n", type=int, default=120000, help="Rows for true-bound proxy.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rep-seed-base", type=int, default=7000)
    parser.add_argument("--maxtime", type=float, default=6.0)
    parser.add_argument("--print-cis", action="store_true")
    args = parser.parse_args()

    b = args.b if args.nsamples is None else args.nsamples

    print(f"Estimating true bounds with true_n={args.true_n} ...", flush=True)
    true_df = simulate_covariate_data(args.true_n, args.seed)
    true_out = solve_point(build_problem(true_df), maxtime=max(args.maxtime, 12.0))
    true_lb = float(true_out["point lb dual"])
    true_ub = float(true_out["point ub dual"])
    print(f"true_lb={true_lb:.8f} true_ub={true_ub:.8f}", flush=True)

    x_levels = sorted(true_df["X"].dropna().unique().tolist())
    true_x = {}
    for x in x_levels:
        tdf_x = true_df.loc[true_df["X"] == x].reset_index(drop=True)
        out_x = solve_point(build_problem_x(tdf_x), maxtime=max(args.maxtime, 12.0))
        true_x[int(x)] = (
            float(out_x["point lb dual"]),
            float(out_x["point ub dual"]),
        )
        print(
            f"true_x={int(x)} lb={true_x[int(x)][0]:.8f} ub={true_x[int(x)][1]:.8f}",
            flush=True,
        )

    cover_lb = cover_ub = cover_joint = 0
    x_cov = {
        int(x): {"lb": 0, "ub": 0, "joint": 0, "n": 0}
        for x in x_levels
    }
    rep_rows = []
    for r in range(args.r):
        df = simulate_covariate_data(args.n, args.rep_seed_base + r)
        out = solve_ci(
            build_problem(df),
            nsamples=b,
            ci_workers=args.ci_workers,
            subsample_rate=args.subsample_rate,
            maxtime=args.maxtime,
        )
        lb = float(out["2.5% lb bounds"])
        ub = float(out["97.5% ub bounds"])
        lb_ok = lb <= true_lb
        ub_ok = ub >= true_ub
        both = lb_ok and ub_ok
        cover_lb += int(lb_ok)
        cover_ub += int(ub_ok)
        cover_joint += int(both)
        rep_rows.append((r, lb, ub, both))

        for x in x_levels:
            df_x = df.loc[df["X"] == x].reset_index(drop=True)
            if df_x.shape[0] == 0:
                continue
            out_x = solve_ci(
                build_problem_x(df_x),
                nsamples=b,
                ci_workers=args.ci_workers,
                subsample_rate=args.subsample_rate,
                maxtime=args.maxtime,
            )
            lb_x = float(out_x["2.5% lb bounds"])
            ub_x = float(out_x["97.5% ub bounds"])
            true_lb_x, true_ub_x = true_x[int(x)]
            lb_ok_x = lb_x <= true_lb_x
            ub_ok_x = ub_x >= true_ub_x
            x_cov[int(x)]["lb"] += int(lb_ok_x)
            x_cov[int(x)]["ub"] += int(ub_ok_x)
            x_cov[int(x)]["joint"] += int(lb_ok_x and ub_ok_x)
            x_cov[int(x)]["n"] += 1

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
    print(f"R={args.r} n={args.n} B={b}")
    print(f"lb_coverage={cover_lb/args.r:.6f}")
    print(f"ub_coverage={cover_ub/args.r:.6f}")
    print(f"joint_coverage={cover_joint/args.r:.6f}")
    print("----- STRATUM (X) COVERAGE -----")
    for x in x_levels:
        s = x_cov[int(x)]
        print(
            f"x={int(x)} "
            f"lb_coverage={_safe_ratio(s['lb'], s['n']):.6f} "
            f"ub_coverage={_safe_ratio(s['ub'], s['n']):.6f} "
            f"joint_coverage={_safe_ratio(s['joint'], s['n']):.6f}"
        )


if __name__ == "__main__":
    main()

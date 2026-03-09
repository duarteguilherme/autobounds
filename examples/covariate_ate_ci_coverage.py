#!/usr/bin/env python3
"""Coverage experiment for ATE bounds with a covariate X."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _one_rep(
    rep_id: int,
    n: int,
    b: int,
    ci_workers: int,
    subsample_rate: float,
    maxtime: float,
    seed_base: int,
    true_lb: float,
    true_ub: float,
    x_levels: list[int],
    true_x: dict[int, tuple[float, float]],
):
    df = simulate_covariate_data(n, seed_base + rep_id)
    out = solve_ci(
        build_problem(df),
        nsamples=b,
        ci_workers=ci_workers,
        subsample_rate=subsample_rate,
        maxtime=maxtime,
    )
    lb = float(out["2.5% lb bounds"])
    ub = float(out["97.5% ub bounds"])
    lb_ok = int(lb <= true_lb)
    ub_ok = int(ub >= true_ub)
    both_ok = int(lb_ok and ub_ok)

    x_results = {}
    for x in x_levels:
        df_x = df.loc[df["X"] == x].reset_index(drop=True)
        if df_x.shape[0] == 0:
            continue
        out_x = solve_ci(
            build_problem_x(df_x),
            nsamples=b,
            ci_workers=ci_workers,
            subsample_rate=subsample_rate,
            maxtime=maxtime,
        )
        lb_x = float(out_x["2.5% lb bounds"])
        ub_x = float(out_x["97.5% ub bounds"])
        true_lb_x, true_ub_x = true_x[int(x)]
        lb_ok_x = int(lb_x <= true_lb_x)
        ub_ok_x = int(ub_x >= true_ub_x)
        x_results[int(x)] = {
            "lb": lb_ok_x,
            "ub": ub_ok_x,
            "joint": int(lb_ok_x and ub_ok_x),
            "n": 1,
        }
    return rep_id, lb, ub, lb_ok, ub_ok, both_ok, x_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=75, help="Number of Monte Carlo datasets.")
    parser.add_argument("--n", type=int, default=600, help="Rows per Monte Carlo dataset.")
    parser.add_argument("--b", type=int, default=500, help="Subsampling reps per dataset.")
    parser.add_argument("--nsamples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
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
    rep_rows = {}
    done = 0
    t0 = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    _one_rep,
                    rep_id=r,
                    n=args.n,
                    b=b,
                    ci_workers=args.ci_workers,
                    subsample_rate=args.subsample_rate,
                    maxtime=args.maxtime,
                    seed_base=args.rep_seed_base,
                    true_lb=true_lb,
                    true_ub=true_ub,
                    x_levels=[int(x) for x in x_levels],
                    true_x=true_x,
                )
                for r in range(args.r)
            ]
            for fut in as_completed(futs):
                rid, lb, ub, lb_i, ub_i, both_i, x_results = fut.result()
                cover_lb += lb_i
                cover_ub += ub_i
                cover_joint += both_i
                rep_rows[rid] = (lb, ub, bool(both_i))
                for x, cov in x_results.items():
                    x_cov[int(x)]["lb"] += int(cov["lb"])
                    x_cov[int(x)]["ub"] += int(cov["ub"])
                    x_cov[int(x)]["joint"] += int(cov["joint"])
                    x_cov[int(x)]["n"] += int(cov["n"])
                done += 1
                if done % 10 == 0 or done == args.r:
                    elapsed = time.monotonic() - t0
                    print(
                        f"progress {done}/{args.r} "
                        f"lb={cover_lb/done:.4f} ub={cover_ub/done:.4f} joint={cover_joint/done:.4f} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
    except PermissionError:
        print("Parallel pool unavailable in this environment. Falling back to serial.", flush=True)
        for r in range(args.r):
            rid, lb, ub, lb_i, ub_i, both_i, x_results = _one_rep(
                rep_id=r,
                n=args.n,
                b=b,
                ci_workers=args.ci_workers,
                subsample_rate=args.subsample_rate,
                maxtime=args.maxtime,
                seed_base=args.rep_seed_base,
                true_lb=true_lb,
                true_ub=true_ub,
                x_levels=[int(x) for x in x_levels],
                true_x=true_x,
            )
            cover_lb += lb_i
            cover_ub += ub_i
            cover_joint += both_i
            rep_rows[rid] = (lb, ub, bool(both_i))
            for x, cov in x_results.items():
                x_cov[int(x)]["lb"] += int(cov["lb"])
                x_cov[int(x)]["ub"] += int(cov["ub"])
                x_cov[int(x)]["joint"] += int(cov["joint"])
                x_cov[int(x)]["n"] += int(cov["n"])
            done += 1
            if done % 10 == 0 or done == args.r:
                elapsed = time.monotonic() - t0
                print(
                    f"progress {done}/{args.r} "
                    f"lb={cover_lb/done:.4f} ub={cover_ub/done:.4f} joint={cover_joint/done:.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if args.print_cis:
        print("----- PER-REPLICATE CIs -----")
        for rid in sorted(rep_rows):
            lb, ub, both = rep_rows[rid]
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

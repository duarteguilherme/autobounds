#!/usr/bin/env python3
"""Coverage experiment for IV ATE bounds CIs with optional parallelization."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autobounds import DAG, causalProblem
from examples.iv_ate_ci_subsampling import simulate_iv_data


@dataclass
class RepResult:
    rep_id: int
    ci_lb: float
    ci_ub: float
    lb_cover: int
    ub_cover: int
    joint_cover: int


def _build_problem(df):
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob="U")
    problem = causalProblem(dag)
    problem.set_ate("X", "Y")
    problem.read_data(raw=df)
    return problem


def _true_bounds(true_n: int, seed: int, maxtime: float) -> Tuple[float, float]:
    df = simulate_iv_data(true_n, seed=seed)
    problem = _build_problem(df)
    out = problem.solve(
        ci=False,
        maxtime=maxtime,
        verbose_optimizer=False,
        verbose_result=False,
        limits=[-1, 1],
    )
    return float(out["point lb dual"]), float(out["point ub dual"])


def _one_rep(
    rep_id: int,
    n: int,
    b: int,
    ci_workers: int,
    ci_method: str,
    subsample_rate: float,
    maxtime: float,
    seed_base: int,
    true_lb: float,
    true_ub: float,
) -> RepResult:
    df = simulate_iv_data(n, seed=seed_base + rep_id)
    problem = _build_problem(df)
    out = problem.solve(
        ci=True,
        nsamples=b,
        ci_workers=ci_workers,
        ci_method=ci_method,
        subsample_rate=subsample_rate,
        maxtime=maxtime,
        verbose_optimizer=False,
        verbose_result=False,
        limits=[-1, 1],
    )
    ci_lb = float(out["2.5% lb bounds"])
    ci_ub = float(out["97.5% ub bounds"])
    lb_cover = int(ci_lb <= true_lb)
    ub_cover = int(ci_ub >= true_ub)
    return RepResult(
        rep_id=rep_id,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        lb_cover=lb_cover,
        ub_cover=ub_cover,
        joint_cover=int(lb_cover and ub_cover),
    )


def _log_progress(done: int, total: int, cover_lb: int, cover_ub: int, cover_joint: int, t0: float) -> None:
    elapsed = time.monotonic() - t0
    pct = (100.0 * done / total) if total else 100.0
    eta = (elapsed / done) * (total - done) if done else float("inf")
    eta_str = f"{eta:.1f}s" if done else "n/a"
    print(
        f"progress {done}/{total} ({pct:5.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta_str} "
        f"lb={cover_lb/done:.4f} ub={cover_ub/done:.4f} joint={cover_joint/done:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=1000, help="Number of Monte Carlo datasets (R).")
    parser.add_argument("--n", type=int, default=600, help="Rows per Monte Carlo dataset (n).")
    parser.add_argument("--b", type=int, default=500, help="Subsampling repetitions per dataset (B).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--ci-workers", type=int, default=16, help="Parallel workers inside each CI solve.")
    parser.add_argument("--subsample-rate", type=float, default=0.7)
    parser.add_argument(
        "--ci-method",
        type=str,
        default="recentered_subsampling",
        choices=["recentered_subsampling", "empirical_subsample_quantile"],
        help="Subsampling CI calibration method.",
    )
    parser.add_argument("--true-n", type=int, default=120000, help="Rows for large-sample true-bound proxy.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rep-seed-base", type=int, default=7000)
    parser.add_argument("--maxtime", type=float, default=4.0, help="Max solver seconds per solve call.")
    parser.add_argument("--progress-every", type=int, default=10, help="Log progress every N completed reps.")
    parser.add_argument("--print-cis", action="store_true", help="Print one CI line per replicate.")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")

    print(f"Estimating true bounds with true_n={args.true_n} ...", flush=True)
    true_lb, true_ub = _true_bounds(args.true_n, args.seed, maxtime=max(args.maxtime, 12.0))
    print(f"true_lb={true_lb:.8f} true_ub={true_ub:.8f}", flush=True)
    print(
        f"Running coverage: R={args.r}, n={args.n}, B={args.b}, workers={args.workers}, "
        f"rate={args.subsample_rate}, method={args.ci_method}, ci_workers={args.ci_workers}",
        flush=True,
    )

    cover_lb = 0
    cover_ub = 0
    cover_joint = 0
    done = 0
    t0 = time.monotonic()
    rep_results = {}

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    _one_rep,
                    rep_id=i,
                    n=args.n,
                    b=args.b,
                    ci_workers=args.ci_workers,
                    ci_method=args.ci_method,
                    subsample_rate=args.subsample_rate,
                    maxtime=args.maxtime,
                    seed_base=args.rep_seed_base,
                    true_lb=true_lb,
                    true_ub=true_ub,
                )
                for i in range(args.r)
            ]
            for fut in as_completed(futures):
                res = fut.result()
                cover_lb += res.lb_cover
                cover_ub += res.ub_cover
                cover_joint += res.joint_cover
                rep_results[res.rep_id] = res
                done += 1
                if done % args.progress_every == 0 or done == args.r:
                    _log_progress(done, args.r, cover_lb, cover_ub, cover_joint, t0)
    except PermissionError:
        print("Parallel pool unavailable in this environment. Falling back to serial.", flush=True)
        for i in range(args.r):
            res = _one_rep(
                rep_id=i,
                n=args.n,
                b=args.b,
                ci_workers=args.ci_workers,
                ci_method=args.ci_method,
                subsample_rate=args.subsample_rate,
                maxtime=args.maxtime,
                seed_base=args.rep_seed_base,
                true_lb=true_lb,
                true_ub=true_ub,
            )
            cover_lb += res.lb_cover
            cover_ub += res.ub_cover
            cover_joint += res.joint_cover
            rep_results[res.rep_id] = res
            done += 1
            if done % args.progress_every == 0 or done == args.r:
                _log_progress(done, args.r, cover_lb, cover_ub, cover_joint, t0)

    if args.print_cis:
        print("----- PER-REPLICATE CIs -----")
        for rep_id in sorted(rep_results):
            rep = rep_results[rep_id]
            print(
                f"rep={rep_id:04d} "
                f"CI=[{rep.ci_lb:.6f}, {rep.ci_ub:.6f}] "
                f"true=[{true_lb:.6f}, {true_ub:.6f}] "
                f"cover={bool(rep.joint_cover)}"
            )

    print("----- FINAL -----")
    print(f"R={args.r} n={args.n} B={args.b}")
    print(f"lb_coverage={cover_lb/args.r:.6f}")
    print(f"ub_coverage={cover_ub/args.r:.6f}")
    print(f"joint_coverage={cover_joint/args.r:.6f}")


if __name__ == "__main__":
    main()

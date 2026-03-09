#!/usr/bin/env python3
"""Coverage experiment for IV ATE bounds without covariates."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autobounds import DAG, causalProblem
from examples.iv_ate_ci_subsampling import simulate_iv_data


def build_problem(df):
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob="U")
    problem = causalProblem(dag)
    problem.set_ate("X", "Y")
    problem.read_data(raw=df)
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
    b: int,
    ci_workers: int,
    subsample_rate: float,
    maxtime: float,
) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return problem.solve(
            ci=True,
            nsamples=b,
            ci_workers=ci_workers,
            subsample_rate=subsample_rate,
            ci_method="recentered_subsampling",
            maxtime=maxtime,
            verbose_optimizer=False,
            verbose_result=False,
            limits=[-1, 1],
        )


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
):
    df = simulate_iv_data(n, seed=seed_base + rep_id)
    out = solve_ci(
        build_problem(df),
        b=b,
        ci_workers=ci_workers,
        subsample_rate=subsample_rate,
        maxtime=maxtime,
    )
    lb = float(out["2.5% lb bounds"])
    ub = float(out["97.5% ub bounds"])
    lb_ok = lb <= true_lb
    ub_ok = ub >= true_ub
    return rep_id, lb, ub, int(lb_ok), int(ub_ok), int(lb_ok and ub_ok)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=75, help="Number of Monte Carlo datasets.")
    parser.add_argument("--n", type=int, default=600, help="Rows per Monte Carlo dataset.")
    parser.add_argument("--b", type=int, default=500, help="Subsampling reps per dataset.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--ci-workers", type=int, default=16, help="Parallel workers inside each CI solve.")
    parser.add_argument("--subsample-rate", type=float, default=(2.0 / 3.0))
    parser.add_argument("--true-n", type=int, default=120000, help="Rows for true-bound proxy.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rep-seed-base", type=int, default=7000)
    parser.add_argument("--maxtime", type=float, default=6.0)
    parser.add_argument("--print-cis", action="store_true")
    args = parser.parse_args()

    print(f"Estimating true bounds with true_n={args.true_n} ...", flush=True)
    true_df = simulate_iv_data(args.true_n, seed=args.seed)
    true_out = solve_point(build_problem(true_df), maxtime=max(args.maxtime, 12.0))
    true_lb = float(true_out["point lb dual"])
    true_ub = float(true_out["point ub dual"])
    print(f"true_lb={true_lb:.8f} true_ub={true_ub:.8f}", flush=True)

    cover_lb = cover_ub = cover_joint = 0
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
                    b=args.b,
                    ci_workers=args.ci_workers,
                    subsample_rate=args.subsample_rate,
                    maxtime=args.maxtime,
                    seed_base=args.rep_seed_base,
                    true_lb=true_lb,
                    true_ub=true_ub,
                )
                for r in range(args.r)
            ]
            for fut in as_completed(futs):
                rid, lb, ub, lb_i, ub_i, both_i = fut.result()
                cover_lb += lb_i
                cover_ub += ub_i
                cover_joint += both_i
                rep_rows[rid] = (lb, ub, bool(both_i))
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
            rid, lb, ub, lb_i, ub_i, both_i = _one_rep(
                rep_id=r,
                n=args.n,
                b=args.b,
                ci_workers=args.ci_workers,
                subsample_rate=args.subsample_rate,
                maxtime=args.maxtime,
                seed_base=args.rep_seed_base,
                true_lb=true_lb,
                true_ub=true_ub,
            )
            cover_lb += lb_i
            cover_ub += ub_i
            cover_joint += both_i
            rep_rows[rid] = (lb, ub, bool(both_i))
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
    print(f"R={args.r} n={args.n} B={args.b}")
    print(f"lb_coverage={cover_lb/args.r:.6f}")
    print(f"ub_coverage={cover_ub/args.r:.6f}")
    print(f"joint_coverage={cover_joint/args.r:.6f}")


if __name__ == "__main__":
    main()

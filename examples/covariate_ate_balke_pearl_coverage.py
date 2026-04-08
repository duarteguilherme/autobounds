#!/usr/bin/env python3
"""Fast coverage experiment using Balke-Pearl binary IV bounds."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


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


def _safe_ratio(num, den):
    return float(num) / float(den) if den > 0 else float("nan")


def bp_iv_bounds(df: pd.DataFrame) -> tuple[float, float]:
    probs = {}
    for z in [0, 1]:
        sub = df.loc[df["Z"] == z]
        if sub.shape[0] == 0:
            raise ValueError("Both instrument levels must be present.")
        for d in [0, 1]:
            for y in [0, 1]:
                probs[(y, d, z)] = float(((sub["D"] == d) & (sub["Y"] == y)).mean())

    p00_0 = probs[(0, 0, 0)]
    p01_0 = probs[(0, 1, 0)]
    p10_0 = probs[(1, 0, 0)]
    p11_0 = probs[(1, 1, 0)]
    p00_1 = probs[(0, 0, 1)]
    p01_1 = probs[(0, 1, 1)]
    p10_1 = probs[(1, 0, 1)]
    p11_1 = probs[(1, 1, 1)]

    lower = max(
        p00_0 + p11_1 - 1.0,
        p00_1 + p11_1 - 1.0,
        p11_0 + p00_1 - 1.0,
        p00_0 + p11_0 - 1.0,
        2.0 * p00_0 + p11_0 + p10_0 + p11_1 - 2.0,
        p00_0 + 2.0 * p11_0 + p00_1 + p01_1 - 2.0,
        p10_0 + p11_0 + 2.0 * p00_1 + p11_1 - 2.0,
        p00_0 + p01_0 + p00_1 + 2.0 * p11_1 - 2.0,
    )
    upper = min(
        1.0 - p10_0 - p01_1,
        1.0 - p01_0 - p10_1,
        1.0 - p01_0 - p10_0,
        1.0 - p01_1 - p10_1,
        2.0 - 2.0 * p01_1 - p10_0 - p10_1 - p11_1,
        2.0 - p01_0 - 2.0 * p10_0 - p00_1 - p01_1,
        2.0 - p10_0 - p11_0 - 2.0 * p01_1 - p10_1,
        2.0 - p00_0 - p01_0 - p01_1 - 2.0 * p10_1,
    )
    return float(lower), float(upper)


def aggregate_covariate_bounds(df: pd.DataFrame) -> tuple[float, float]:
    total = float(df.shape[0])
    lb = 0.0
    ub = 0.0
    for _, gdf in df.groupby("X", sort=False, dropna=False):
        w = float(gdf.shape[0]) / total
        g_lb, g_ub = bp_iv_bounds(gdf[["Z", "D", "Y"]].reset_index(drop=True))
        lb += w * g_lb
        ub += w * g_ub
    return lb, ub


def _resolve_subsample_size(n: int, subsample_rate: float, subsample_size: int | None) -> int:
    if subsample_size is None:
        return min(n, max(1, int(np.floor(n ** subsample_rate))))
    return min(n, int(subsample_size))


def _allocate_strata_counts(counts: np.ndarray, m: int) -> np.ndarray:
    n = int(counts.sum())
    if m >= n:
        return counts.copy()
    weights = counts / n
    raw = m * weights
    alloc = np.floor(raw).astype(int)
    positive = counts > 0
    for idx in np.where((alloc == 0) & positive)[0]:
        alloc[idx] = 1
    alloc = np.minimum(alloc, counts)
    total = int(alloc.sum())
    if total > m:
        excess = total - m
        removable = np.argsort(-(alloc - 1))
        for idx in removable:
            if excess == 0:
                break
            drop = min(excess, max(0, alloc[idx] - 1))
            alloc[idx] -= drop
            excess -= drop
    elif total < m:
        deficit = m - total
        room = counts - alloc
        add_order = np.argsort(-(raw - np.floor(raw)))
        while deficit > 0 and np.any(room > 0):
            progressed = False
            for idx in add_order:
                if room[idx] <= 0 or deficit == 0:
                    continue
                alloc[idx] += 1
                room[idx] -= 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
            if not progressed:
                break
    return alloc


def stratified_subsample(
    df: pd.DataFrame,
    covariates: list[str],
    subsample_rate: float,
    subsample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    if covariates is None or len(covariates) == 0:
        m = _resolve_subsample_size(int(df.shape[0]), subsample_rate, subsample_size)
        return df.sample(n=m, replace=False, random_state=random_state).reset_index(drop=True)

    m = _resolve_subsample_size(int(df.shape[0]), subsample_rate, subsample_size)
    grouped = list(df.groupby(covariates, sort=False, dropna=False))
    counts = np.array([g.shape[0] for _, g in grouped], dtype=int)
    alloc = _allocate_strata_counts(counts, m)
    rng = np.random.default_rng(random_state)
    pieces = []
    for (_, gdf), take in zip(grouped, alloc):
        if take <= 0:
            continue
        pieces.append(
            gdf.sample(n=int(take), replace=False, random_state=int(rng.integers(0, 2**32 - 1)))
        )
    return (
        pd.concat(pieces, axis=0)
        .sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1)))
        .reset_index(drop=True)
    )


def recentered_ci(
    solve_fn,
    df: pd.DataFrame,
    b: int,
    subsample_rate: float,
    ci_workers: int,
    strat_cols: list[str] | None = None,
    subsample_size: int | None = None,
    rep_seeds: list[int] | np.ndarray | None = None,
    subsample_dfs: list[pd.DataFrame] | None = None,
) -> dict[str, float]:
    theta_lb, theta_ub = solve_fn(df)
    n = int(df.shape[0])
    m = _resolve_subsample_size(n, subsample_rate, subsample_size)
    if subsample_dfs is not None:
        if len(subsample_dfs) != b:
            raise ValueError("subsample_dfs length must match b.")
        sample_dfs = subsample_dfs
    else:
        if rep_seeds is None:
            rep_seeds = np.random.default_rng().integers(0, 2**32 - 1, size=b)
        else:
            rep_seeds = np.asarray(rep_seeds, dtype=np.int64)
            if rep_seeds.ndim != 1:
                raise ValueError("rep_seeds must be a one-dimensional array-like.")
            if rep_seeds.size != b:
                raise ValueError("rep_seeds length must match b.")
        sample_dfs = [
            stratified_subsample(df, strat_cols or [], subsample_rate, subsample_size, int(seed))
            for seed in rep_seeds
        ]

    def _one(sub_df):
        return solve_fn(sub_df)

    if ci_workers > 1:
        with ThreadPoolExecutor(max_workers=ci_workers) as ex:
            samples = list(ex.map(_one, sample_dfs))
    else:
        samples = [_one(sub_df) for sub_df in sample_dfs]

    lb_arr = np.asarray([x[0] for x in samples], dtype=float)
    ub_arr = np.asarray([x[1] for x in samples], dtype=float)
    t_lb = np.sqrt(m) * (lb_arr - theta_lb)
    t_ub = np.sqrt(m) * (ub_arr - theta_ub)
    sqrt_n = np.sqrt(n)
    return {
        "point lb dual": float(theta_lb),
        "point ub dual": float(theta_ub),
        "2.5% lb bounds": float(theta_lb - np.quantile(t_lb, 0.975) / sqrt_n),
        "97.5% ub bounds": float(theta_ub - np.quantile(t_ub, 0.025) / sqrt_n),
        "1% lb bounds": float(theta_lb - np.quantile(t_lb, 0.99) / sqrt_n),
        "99% ub bounds": float(theta_ub - np.quantile(t_ub, 0.01) / sqrt_n),
    }


def _one_rep(
    rep_id: int,
    n: int,
    b: int,
    ci_workers: int,
    subsample_rate: float,
    seed_base: int,
    true_lb: float,
    true_ub: float,
    x_levels: list[int],
    true_x: dict[int, tuple[float, float]],
):
    df = simulate_covariate_data(n, seed_base + rep_id)
    out = recentered_ci(
        aggregate_covariate_bounds,
        df,
        b=b,
        subsample_rate=subsample_rate,
        ci_workers=ci_workers,
    )
    lb = float(out["2.5% lb bounds"])
    ub = float(out["97.5% ub bounds"])
    lb_ok = int(lb <= true_lb)
    ub_ok = int(ub >= true_ub)
    both_ok = int(lb_ok and ub_ok)

    x_results = {}
    for x in x_levels:
        df_x = df.loc[df["X"] == x, ["Z", "D", "Y"]].reset_index(drop=True)
        out_x = recentered_ci(
            bp_iv_bounds,
            df_x,
            b=b,
            subsample_rate=subsample_rate,
            ci_workers=ci_workers,
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
    parser.add_argument("--n", type=int, default=1500, help="Rows per Monte Carlo dataset.")
    parser.add_argument("--b", type=int, default=500, help="Subsampling reps per dataset.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--ci-workers", type=int, default=16, help="Parallel workers inside each CI solve.")
    parser.add_argument("--subsample-rate", type=float, default=(2.0 / 3.0))
    parser.add_argument("--true-n", type=int, default=120000, help="Rows for true-bound proxy.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rep-seed-base", type=int, default=7000)
    parser.add_argument("--print-cis", action="store_true")
    args = parser.parse_args()

    print(f"Estimating true bounds with true_n={args.true_n} ...", flush=True)
    true_df = simulate_covariate_data(args.true_n, args.seed)
    true_lb, true_ub = aggregate_covariate_bounds(true_df)
    print(f"true_lb={true_lb:.8f} true_ub={true_ub:.8f}", flush=True)

    x_levels = sorted(int(x) for x in true_df["X"].dropna().unique().tolist())
    true_x = {}
    for x in x_levels:
        true_x[int(x)] = bp_iv_bounds(true_df.loc[true_df["X"] == x, ["Z", "D", "Y"]].reset_index(drop=True))
        print(f"true_x={int(x)} lb={true_x[int(x)][0]:.8f} ub={true_x[int(x)][1]:.8f}", flush=True)

    cover_lb = cover_ub = cover_joint = 0
    x_cov = {int(x): {"lb": 0, "ub": 0, "joint": 0, "n": 0} for x in x_levels}
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
                    seed_base=args.rep_seed_base,
                    true_lb=true_lb,
                    true_ub=true_ub,
                    x_levels=x_levels,
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
                b=args.b,
                ci_workers=args.ci_workers,
                subsample_rate=args.subsample_rate,
                seed_base=args.rep_seed_base,
                true_lb=true_lb,
                true_ub=true_ub,
                x_levels=x_levels,
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
    print(f"R={args.r} n={args.n} B={args.b}")
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

#!/usr/bin/env python3
"""Joint CI comparison for the Section 7 IV ATE model in g2.pdf.

This script uses a calibrated principal-strata DGP matching the paper's
reported ATE setup in Figure 5(b): exclusion holds, monotonicity fails,
and the population ATE bounds are approximately [-0.5502, -0.1460].
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from balke_pearl_closed_form_ci_compare import (
    LOWER_COEFFS,
    LOWER_CONSTS,
    UPPER_COEFFS,
    UPPER_CONSTS,
    candidate_standard_error,
    bayesian_dirichlet_lower_endpoint,
    bayesian_dirichlet_upper_endpoint,
    bootstrap_lower_endpoint,
    bootstrap_upper_endpoint,
    mtn_kl_lower_endpoint,
    mtn_kl_upper_endpoint,
    recentered_subsampling_lower_endpoint,
    recentered_subsampling_upper_endpoint,
)

X_TYPES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
Y_TYPES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
NORMAL_975 = 1.959963984540054

# Joint distribution over (X-response-type, Y-response-type).
# Rows: never-taker, complier, defier, always-taker.
# Cols: Y-never, Y-positive, Y-negative, Y-always.
# Calibrated to: true ATE=-0.25 and Balke-Pearl ATE bounds ~= [-0.5502, -0.1460].
JOINT_STRATA = np.asarray(
    [
        [0.01263684, 0.00941365, 0.00716565, 0.07256184],
        [0.02938684, 0.03262841, 0.26587725, 0.32002131],
        [0.01265782, 0.01096552, 0.02152598, 0.00696449],
        [0.16663137, 0.01004004, 0.01847827, 0.00304472],
    ],
    dtype=float,
)
JOINT_STRATA = JOINT_STRATA / JOINT_STRATA.sum()


@dataclass
class RepResult:
    point_lb: float
    point_ub: float
    oracle_lb: float
    oracle_ub: float
    bootstrap_lb: float
    bootstrap_ub: float
    subsample_lb: float
    subsample_ub: float
    mtn_kl_lb: float
    mtn_kl_ub: float
    bayes_dirichlet_lb: float
    bayes_dirichlet_ub: float


def obs_probs_from_joint(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(4, 4)
    out = []
    for z in [0, 1]:
        xz = X_TYPES[:, z]
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            s = 0.0
            for i in range(4):
                xval = xz[i]
                for j in range(4):
                    yval = Y_TYPES[j, xval]
                    if xval == x and yval == y:
                        s += p[i, j]
            out.append(s)
    return np.asarray(out, dtype=float)


def lower_candidates_from_probs(probs: np.ndarray) -> np.ndarray:
    return LOWER_COEFFS @ probs + LOWER_CONSTS


def upper_candidates_from_probs(probs: np.ndarray) -> np.ndarray:
    return UPPER_COEFFS @ probs + UPPER_CONSTS


def empirical_conditional_probs(data: np.ndarray) -> np.ndarray:
    z = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    out = np.empty(8, dtype=float)
    pos = 0
    for z_level in [0, 1]:
        mask_z = z == z_level
        if int(mask_z.sum()) == 0:
            raise ValueError("Both instrument levels must be present.")
        for x_level, y_level in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            out[pos] = np.mean(mask_z & (x == x_level) & (y == y_level)) / np.mean(mask_z)
            pos += 1
    return out


def lower_bound_from_data(data: np.ndarray) -> tuple[float, int, np.ndarray]:
    probs = empirical_conditional_probs(data)
    candidates = lower_candidates_from_probs(probs)
    idx = int(np.argmax(candidates))
    return float(candidates[idx]), idx, candidates


def upper_bound_from_data(data: np.ndarray) -> tuple[float, int, np.ndarray]:
    probs = empirical_conditional_probs(data)
    candidates = upper_candidates_from_probs(probs)
    idx = int(np.argmin(candidates))
    return float(candidates[idx]), idx, candidates


def simulate_data(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = JOINT_STRATA.reshape(-1)
    idx = rng.choice(flat.size, size=n, p=flat)
    x_type = idx // 4
    y_type = idx % 4
    z = rng.binomial(1, 0.5, size=n)
    x = X_TYPES[x_type, z]
    y = Y_TYPES[y_type, x]
    return np.column_stack([z, x, y]).astype(np.int8, copy=False)


def true_quantities() -> dict[str, float | int | np.ndarray]:
    probs = obs_probs_from_joint(JOINT_STRATA)
    lb_candidates = lower_candidates_from_probs(probs)
    ub_candidates = upper_candidates_from_probs(probs)
    lb = float(np.max(lb_candidates))
    ub = float(np.min(ub_candidates))
    oracle_lb_idx = int(np.argmax(lb_candidates))
    oracle_ub_idx = int(np.argmin(ub_candidates))
    ate = float(np.sum(JOINT_STRATA * (Y_TYPES[:, 1] - Y_TYPES[:, 0])[None, :]))
    return {
        "lb": lb,
        "ub": ub,
        "ate": ate,
        "oracle_lb_idx": oracle_lb_idx,
        "oracle_ub_idx": oracle_ub_idx,
        "obs_probs": probs,
    }


def run_one_rep(n: int, b: int, alpha: float, gamma: float, seed: int) -> RepResult:
    truth = true_quantities()
    oracle_lb_idx = int(truth["oracle_lb_idx"])
    oracle_ub_idx = int(truth["oracle_ub_idx"])

    data = simulate_data(n=n, seed=seed)
    point_lb, _, sample_lb_candidates = lower_bound_from_data(data)
    point_ub, _, sample_ub_candidates = upper_bound_from_data(data)

    oracle_lb_est = float(sample_lb_candidates[oracle_lb_idx])
    oracle_lb = float(oracle_lb_est - NORMAL_975 * candidate_standard_error(data, coeffs=LOWER_COEFFS[oracle_lb_idx]))
    oracle_ub_est = float(sample_ub_candidates[oracle_ub_idx])
    oracle_ub = float(oracle_ub_est + NORMAL_975 * candidate_standard_error(data, coeffs=UPPER_COEFFS[oracle_ub_idx]))

    bootstrap_lb = bootstrap_lower_endpoint(data, b=b, alpha=alpha, seed=seed + 10_000_000)
    bootstrap_ub = bootstrap_upper_endpoint(data, b=b, alpha=alpha, seed=seed + 15_000_000)
    subsample_lb = recentered_subsampling_lower_endpoint(data, b=b, alpha=alpha, gamma=gamma, seed=seed + 20_000_000)
    subsample_ub = recentered_subsampling_upper_endpoint(data, b=b, alpha=alpha, gamma=gamma, seed=seed + 25_000_000)
    mtn_kl_lb = mtn_kl_lower_endpoint(data, alpha=alpha, seed=seed + 30_000_000)
    mtn_kl_ub = mtn_kl_upper_endpoint(data, alpha=alpha, seed=seed + 35_000_000)
    bayes_dirichlet_lb = bayesian_dirichlet_lower_endpoint(data, b=b, alpha=alpha, seed=seed + 40_000_000)
    bayes_dirichlet_ub = bayesian_dirichlet_upper_endpoint(data, b=b, alpha=alpha, seed=seed + 45_000_000)

    return RepResult(
        point_lb=point_lb,
        point_ub=point_ub,
        oracle_lb=oracle_lb,
        oracle_ub=oracle_ub,
        bootstrap_lb=bootstrap_lb,
        bootstrap_ub=bootstrap_ub,
        subsample_lb=subsample_lb,
        subsample_ub=subsample_ub,
        mtn_kl_lb=mtn_kl_lb,
        mtn_kl_ub=mtn_kl_ub,
        bayes_dirichlet_lb=bayes_dirichlet_lb,
        bayes_dirichlet_ub=bayes_dirichlet_ub,
    )


def summarize_results(results: list[RepResult], true_lb: float, true_ub: float) -> dict[str, float]:
    point_lbs = np.asarray([res.point_lb for res in results], dtype=float)
    point_ubs = np.asarray([res.point_ub for res in results], dtype=float)
    methods = {
        "oracle": (np.asarray([r.oracle_lb for r in results], float), np.asarray([r.oracle_ub for r in results], float)),
        "bootstrap": (np.asarray([r.bootstrap_lb for r in results], float), np.asarray([r.bootstrap_ub for r in results], float)),
        "subsample": (np.asarray([r.subsample_lb for r in results], float), np.asarray([r.subsample_ub for r in results], float)),
        "mtn_kl": (np.asarray([r.mtn_kl_lb for r in results], float), np.asarray([r.mtn_kl_ub for r in results], float)),
        "bayes_dirichlet": (np.asarray([r.bayes_dirichlet_lb for r in results], float), np.asarray([r.bayes_dirichlet_ub for r in results], float)),
    }
    out = {"mean_point_lb": float(point_lbs.mean()), "mean_point_ub": float(point_ubs.mean())}
    for name, (lbs, ubs) in methods.items():
        out[f"{name}_joint_coverage"] = float(np.mean((lbs <= true_lb) & (ubs >= true_ub)))
        out[f"{name}_joint_avg_width"] = float(np.mean(ubs - lbs))
    return out


def save_summary_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    fields = [
        "n", "mean_point_lb", "mean_point_ub",
        "oracle_joint_coverage", "oracle_joint_avg_width",
        "bootstrap_joint_coverage", "bootstrap_joint_avg_width",
        "subsample_joint_coverage", "subsample_joint_avg_width",
        "mtn_kl_joint_coverage", "mtn_kl_joint_avg_width",
        "bayes_dirichlet_joint_coverage", "bayes_dirichlet_joint_avg_width",
    ]
    with out_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plots(rows: list[dict[str, float]], out_dir: Path, stem: str) -> list[Path]:
    n_vals = np.asarray([int(row["n"]) for row in rows], dtype=int)
    methods = [
        ("oracle", "Oracle", "o"),
        ("bootstrap", "Bootstrap", "s"),
        ("subsample", "Subsampling", "^"),
        ("mtn_kl", "MTN KL", "d"),
        ("bayes_dirichlet", "Bayes Dirichlet", "x"),
    ]
    saved = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, marker in methods:
        vals = np.asarray([float(row[f"{key}_joint_coverage"]) for row in rows], dtype=float)
        ax.plot(n_vals, vals, marker=marker, linewidth=2, label=label)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_ylim(0.6, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.set_title("Section 7 IV ATE: Joint CI Coverage")
    ax.set_xlabel("n")
    ax.set_ylabel("Coverage")
    ax.grid(alpha=0.25)
    ax.legend()
    path = out_dir / f"{stem}_joint_coverage.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, marker in methods:
        vals = np.asarray([float(row[f"{key}_joint_avg_width"]) for row in rows], dtype=float)
        ax.plot(n_vals, vals, marker=marker, linewidth=2, label=label)
    ax.set_title("Section 7 IV ATE: Joint CI Width")
    ax.set_xlabel("n")
    ax.set_ylabel("Average Width")
    ax.grid(alpha=0.25)
    ax.legend()
    path = out_dir / f"{stem}_joint_width.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    return saved


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "n/a"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=1000)
    parser.add_argument("--b", type=int, default=500)
    parser.add_argument("--n-grid", type=str, default="100,200,300,500,1000,2000")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--subsample-gamma", type=float, default=(2.0 / 3.0))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/outputs"))
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_grid = [int(x.strip()) for x in args.n_grid.split(",") if x.strip()]
    truth = true_quantities()
    stem = "paper_iv_ate_joint_ci"

    print(f"true_lower_bound={truth['lb']:.8f}")
    print(f"true_upper_bound={truth['ub']:.8f}")
    print(f"true_ate={truth['ate']:.8f}")
    print(f"oracle_lower_candidate={int(truth['oracle_lb_idx']) + 1}")
    print(f"oracle_upper_candidate={int(truth['oracle_ub_idx']) + 1}")
    print("observed_pzyx=" + ",".join(f"{x:.8f}" for x in truth['obs_probs']))
    print("methods=oracle,bootstrap,subsample,mtn_kl,bayes_dirichlet")
    print(f"subsampling_rule=m=min(n,max(1,floor(n**{args.subsample_gamma:.6f})))")

    spawned = np.random.SeedSequence(args.seed).spawn(len(n_grid))
    summary_rows = []
    overall_t0 = time.monotonic()
    total_jobs = len(n_grid) * args.r
    completed_jobs = 0
    for n in n_grid:
        t0 = time.monotonic()
        rep_seeds = spawned[len(summary_rows)].generate_state(args.r, dtype=np.uint64)
        print(f"starting_n={n} reps={args.r} resamples={args.b} completed_jobs={completed_jobs}/{total_jobs}", flush=True)
        results = []
        if args.workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = [
                        ex.submit(run_one_rep, n=n, b=args.b, alpha=args.alpha, gamma=args.subsample_gamma, seed=int(rep_seeds[i]))
                        for i in range(args.r)
                    ]
                    done = 0
                    for fut in as_completed(futs):
                        results.append(fut.result())
                        done += 1
                        completed_jobs += 1
                        if done % args.progress_every == 0 or done == args.r:
                            elapsed = time.monotonic() - overall_t0
                            rate = completed_jobs / elapsed if elapsed > 0 else float('nan')
                            remaining = (total_jobs - completed_jobs) / rate if rate and np.isfinite(rate) else float('nan')
                            print(f"progress n={n} rep={done}/{args.r} overall={completed_jobs}/{total_jobs} elapsed={format_eta(elapsed)} eta={format_eta(remaining)}", flush=True)
            except PermissionError:
                for i in range(args.r):
                    results.append(run_one_rep(n=n, b=args.b, alpha=args.alpha, gamma=args.subsample_gamma, seed=int(rep_seeds[i])))
                    completed_jobs += 1
        else:
            for i in range(args.r):
                results.append(run_one_rep(n=n, b=args.b, alpha=args.alpha, gamma=args.subsample_gamma, seed=int(rep_seeds[i])))
                completed_jobs += 1
                if (i + 1) % args.progress_every == 0 or (i + 1) == args.r:
                    elapsed = time.monotonic() - overall_t0
                    rate = completed_jobs / elapsed if elapsed > 0 else float('nan')
                    remaining = (total_jobs - completed_jobs) / rate if rate and np.isfinite(rate) else float('nan')
                    print(f"progress n={n} rep={i+1}/{args.r} overall={completed_jobs}/{total_jobs} elapsed={format_eta(elapsed)} eta={format_eta(remaining)}", flush=True)

        summary = summarize_results(results, true_lb=float(truth['lb']), true_ub=float(truth['ub']))
        row = {"n": n, **summary}
        summary_rows.append(row)
        print(
            f"{n},{summary['mean_point_lb']:.6f},{summary['mean_point_ub']:.6f},"
            f"{summary['oracle_joint_coverage']:.4f},{summary['oracle_joint_avg_width']:.6f},"
            f"{summary['bootstrap_joint_coverage']:.4f},{summary['bootstrap_joint_avg_width']:.6f},"
            f"{summary['subsample_joint_coverage']:.4f},{summary['subsample_joint_avg_width']:.6f},"
            f"{summary['mtn_kl_joint_coverage']:.4f},{summary['mtn_kl_joint_avg_width']:.6f},"
            f"{summary['bayes_dirichlet_joint_coverage']:.4f},{summary['bayes_dirichlet_joint_avg_width']:.6f}",
            flush=True,
        )
        print(f"elapsed_for_n={n}:{time.monotonic() - t0:.1f}s", flush=True)

    csv_path = out_dir / f"{stem}.csv"
    save_summary_csv(summary_rows, csv_path)
    plot_paths = save_plots(summary_rows, out_dir, stem)
    print(f"csv_path={csv_path}")
    for path in plot_paths:
        print(f"plot_path={path}")


if __name__ == "__main__":
    main()

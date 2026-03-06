import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SimulationConfig:
    true_means: np.ndarray
    n_values: list
    reps: int
    boot_reps: int
    mc_reps: int
    alpha: float
    seed: int


def compute_oracle_ci(sample_means, sample_sds, n, true_min_index, alpha):
    se = sample_sds[true_min_index] / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    center = sample_means[true_min_index]
    half_width = t_crit * se
    return center - half_width, center + half_width


def bootstrap_ci(samples, n, boot_reps, alpha, rng):
    boot_stats = np.empty(boot_reps)
    for b in range(boot_reps):
        boot_means = np.array([
            rng.choice(group, size=n, replace=True).mean() for group in samples
        ])
        boot_stats[b] = boot_means.min()
    lower = np.quantile(boot_stats, alpha / 2)
    upper = np.quantile(boot_stats, 1 - alpha / 2)
    return lower, upper


def numerical_bootstrap_ci(samples, sample_means, n, boot_reps, alpha, rng, eps=None):
    if eps is None:
        eps = n ** -0.25
    theta_hat = sample_means.min()
    sqrt_n = math.sqrt(n)
    boot_stats = np.empty(boot_reps)
    for b in range(boot_reps):
        boot_means = np.array([
            rng.choice(group, size=n, replace=True).mean() for group in samples
        ])
        perturbed_means = sample_means + eps * sqrt_n * (boot_means - sample_means)
        theta_perturb = perturbed_means.min()
        boot_stats[b] = (theta_perturb - theta_hat) / (eps**2)
    q_lo = np.quantile(boot_stats, alpha / 2)
    q_hi = np.quantile(boot_stats, 1 - alpha / 2)
    lower = theta_hat - q_hi / sqrt_n
    upper = theta_hat - q_lo / sqrt_n
    return lower, upper


def subsampling_ci(samples, sample_means, n, sub_reps, alpha, rng):
    b = max(2, min(n, int(math.ceil(math.log(n)))))
    theta_hat = sample_means.min()
    stats_sub = np.empty(sub_reps)
    for s in range(sub_reps):
        sub_means = np.array([
            rng.choice(group, size=b, replace=False).mean() for group in samples
        ])
        theta_sub = sub_means.min()
        stats_sub[s] = math.sqrt(b) * (theta_sub - theta_hat)
    q_lo = np.quantile(stats_sub, alpha / 2)
    q_hi = np.quantile(stats_sub, 1 - alpha / 2)
    lower = theta_hat - q_hi / math.sqrt(n)
    upper = theta_hat - q_lo / math.sqrt(n)
    return lower, upper


def subsampling_larger_schedule_ci(samples, sample_means, n, sub_reps, alpha, rng):
    b = max(2, min(n, int(math.floor(n**0.6))))
    theta_hat = sample_means.min()
    stats_sub = np.empty(sub_reps)
    for s in range(sub_reps):
        sub_means = np.array([
            rng.choice(group, size=b, replace=False).mean() for group in samples
        ])
        theta_sub = sub_means.min()
        stats_sub[s] = math.sqrt(b) * (theta_sub - theta_hat)
    q_lo = np.quantile(stats_sub, alpha / 2)
    q_hi = np.quantile(stats_sub, 1 - alpha / 2)
    lower = theta_hat - q_hi / math.sqrt(n)
    upper = theta_hat - q_lo / math.sqrt(n)
    return lower, upper


def min_normal_quantiles(se, binding_idx, alpha, mc_reps, rng):
    se = se[binding_idx]
    if se.size == 1:
        q_lo = stats.norm.ppf(alpha / 2, scale=se[0])
        q_hi = stats.norm.ppf(1 - alpha / 2, scale=se[0])
        return q_lo, q_hi
    draws = rng.normal(size=(mc_reps, se.size)) * se
    mins = draws.min(axis=1)
    q_lo = np.quantile(mins, alpha / 2)
    q_hi = np.quantile(mins, 1 - alpha / 2)
    return q_lo, q_hi


def fs_ci(sample_means, sample_sds, n, alpha, mc_reps, rng, kappa):
    theta_hat = sample_means.min()
    se = np.nan_to_num(sample_sds / math.sqrt(n), nan=0.0, posinf=0.0, neginf=0.0)
    gaps = sample_means - theta_hat
    binding_idx = np.where(gaps <= kappa * se)[0]
    if binding_idx.size == 0:
        binding_idx = np.array([int(np.argmin(sample_means))])
    q_lo, q_hi = min_normal_quantiles(se, binding_idx, alpha, mc_reps, rng)
    lower = theta_hat - q_hi
    upper = theta_hat - q_lo
    return lower, upper


def simulate_config(config: SimulationConfig):
    rng = np.random.default_rng(config.seed)
    records = []
    true_min_index = int(np.argmin(config.true_means))
    true_theta = float(config.true_means.min())

    for n in config.n_values:
        kappa = math.sqrt(math.log(n))
        for _ in range(config.reps):
            samples = [
                rng.normal(loc=mu, scale=1.0, size=n) for mu in config.true_means
            ]
            sample_means = np.array([sample.mean() for sample in samples])
            sample_sds = np.array([sample.std(ddof=1) for sample in samples])

            oracle_ci = compute_oracle_ci(
                sample_means, sample_sds, n, true_min_index, config.alpha
            )
            bootstrap_ci_interval = bootstrap_ci(
                samples, n, config.boot_reps, config.alpha, rng
            )
            numerical_ci_interval = numerical_bootstrap_ci(
                samples, sample_means, n, config.boot_reps, config.alpha, rng
            )
            subsampling_ci_interval = subsampling_ci(
                samples, sample_means, n, config.boot_reps, config.alpha, rng
            )
            subsampling_larger_schedule_interval = subsampling_larger_schedule_ci(
                samples, sample_means, n, config.boot_reps, config.alpha, rng
            )
            lf_ci = fs_ci(
                sample_means,
                sample_sds,
                n,
                config.alpha,
                config.mc_reps,
                rng,
                kappa=np.inf,
            )
            fs_ci_interval = fs_ci(
                sample_means,
                sample_sds,
                n,
                config.alpha,
                config.mc_reps,
                rng,
                kappa=kappa,
            )

            for method, (lower, upper) in (
                ("Oracle", oracle_ci),
                ("Bootstrap", bootstrap_ci_interval),
                ("Numerical bootstrap", numerical_ci_interval),
                ("Subsampling", subsampling_ci_interval),
                ("Subsampling-larger-schedule", subsampling_larger_schedule_interval),
                ("Least-favorable", lf_ci),
                ("Fang-Santos", fs_ci_interval),
            ):
                records.append(
                    {
                        "n": n,
                        "method": method,
                        "lower": lower,
                        "upper": upper,
                        "covers": lower <= true_theta <= upper,
                        "length": upper - lower,
                    }
                )

    return pd.DataFrame.from_records(records)


def plot_results(df, output_dir, prefix=None):
    os.makedirs(output_dir, exist_ok=True)
    summary = (
        df.groupby(["n", "method"], as_index=False)
        .agg(coverage=("covers", "mean"), avg_length=("length", "mean"))
    )
    prefix = f"{prefix}-" if prefix else ""

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary["method"].unique():
        subset = summary[summary["method"] == method]
        ax.plot(subset["n"], subset["coverage"], marker="o", label=method)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Coverage")
    ax.set_title("CI Coverage for min(mu)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}coverage.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary["method"].unique():
        subset = summary[summary["method"] == method]
        ax.plot(subset["n"], subset["avg_length"], marker="o", label=method)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Average CI length")
    ax.set_title("CI Length for min(mu)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}length.png"), dpi=150)
    plt.close(fig)

    summary.to_csv(os.path.join(output_dir, f"{prefix}summary.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate CIs for the minimum of five means."
    )
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--boot-reps", type=int, default=300)
    parser.add_argument("--mc-reps", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SimulationConfig(
        true_means=np.array([0.0, 0.01, 0.02, 0.03, 0.04]),
        n_values=[50, 100, 200, 500, 1000, 5000, 10000],
        reps=args.reps,
        boot_reps=args.boot_reps,
        mc_reps=args.mc_reps,
        alpha=args.alpha,
        seed=args.seed,
    )
    df = simulate_config(config)
    plot_results(df, args.output)


if __name__ == "__main__":
    main()

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from simulation import SimulationConfig


def compute_oracle_lower(sample_means, sample_sds, n, true_max_index, alpha):
    se = sample_sds[true_max_index] / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, df=n - 1)
    center = sample_means[true_max_index]
    return center - t_crit * se


def bootstrap_lower(samples, n, boot_reps, alpha, rng):
    boot_stats = np.empty(boot_reps)
    for b in range(boot_reps):
        boot_means = np.array([
            rng.choice(group, size=n, replace=True).mean() for group in samples
        ])
        boot_stats[b] = boot_means.max()
    return np.quantile(boot_stats, alpha)


def numerical_bootstrap_lower(samples, sample_means, n, boot_reps, alpha, rng, eps=None):
    if eps is None:
        eps = n ** -0.25
    theta_hat = sample_means.max()
    sqrt_n = math.sqrt(n)
    boot_stats = np.empty(boot_reps)
    for b in range(boot_reps):
        boot_means = np.array([
            rng.choice(group, size=n, replace=True).mean() for group in samples
        ])
        perturbed_means = sample_means + eps * sqrt_n * (boot_means - sample_means)
        theta_perturb = perturbed_means.max()
        boot_stats[b] = (theta_perturb - theta_hat) / (eps**2)
    q_hi = np.quantile(boot_stats, 1 - alpha)
    return theta_hat - q_hi / sqrt_n


def subsampling_lower(samples, sample_means, n, sub_reps, alpha, rng):
    b = max(2, min(n, int(math.ceil(math.log(n)))))
    theta_hat = sample_means.max()
    stats_sub = np.empty(sub_reps)
    for s in range(sub_reps):
        sub_means = np.array([
            rng.choice(group, size=b, replace=False).mean() for group in samples
        ])
        theta_sub = sub_means.max()
        stats_sub[s] = math.sqrt(b) * (theta_sub - theta_hat)
    q_hi = np.quantile(stats_sub, 1 - alpha)
    return theta_hat - q_hi / math.sqrt(n)


def subsampling_larger_schedule_lower(samples, sample_means, n, sub_reps, alpha, rng):
    b = max(2, min(n, int(math.floor(n**0.6))))
    theta_hat = sample_means.max()
    stats_sub = np.empty(sub_reps)
    for s in range(sub_reps):
        sub_means = np.array([
            rng.choice(group, size=b, replace=False).mean() for group in samples
        ])
        theta_sub = sub_means.max()
        stats_sub[s] = math.sqrt(b) * (theta_sub - theta_hat)
    q_hi = np.quantile(stats_sub, 1 - alpha)
    return theta_hat - q_hi / math.sqrt(n)


def max_normal_quantile(se, binding_idx, alpha, mc_reps, rng):
    se = se[binding_idx]
    if se.size == 1:
        return stats.norm.ppf(1 - alpha, scale=se[0])
    draws = rng.normal(size=(mc_reps, se.size)) * se
    maxes = draws.max(axis=1)
    return np.quantile(maxes, 1 - alpha)


def fs_lower(sample_means, sample_sds, n, alpha, mc_reps, rng, kappa):
    theta_hat = sample_means.max()
    se = np.nan_to_num(sample_sds / math.sqrt(n), nan=0.0, posinf=0.0, neginf=0.0)
    gaps = theta_hat - sample_means
    binding_idx = np.where(gaps <= kappa * se)[0]
    if binding_idx.size == 0:
        binding_idx = np.array([int(np.argmax(sample_means))])
    q_hi = max_normal_quantile(se, binding_idx, alpha, mc_reps, rng)
    return theta_hat - q_hi


def simulate_config_one_sided(config: SimulationConfig):
    rng = np.random.default_rng(config.seed)
    records = []
    true_max_index = int(np.argmax(config.true_means))
    true_theta = float(config.true_means.max())

    for n in config.n_values:
        kappa = math.sqrt(math.log(n))
        for _ in range(config.reps):
            samples = [
                rng.normal(loc=mu, scale=1.0, size=n) for mu in config.true_means
            ]
            sample_means = np.array([sample.mean() for sample in samples])
            sample_sds = np.array([sample.std(ddof=1) for sample in samples])

            oracle_lower = compute_oracle_lower(
                sample_means, sample_sds, n, true_max_index, config.alpha
            )
            bootstrap_lower_bound = bootstrap_lower(
                samples, n, config.boot_reps, config.alpha, rng
            )
            numerical_lower_bound = numerical_bootstrap_lower(
                samples, sample_means, n, config.boot_reps, config.alpha, rng
            )
            subsampling_lower_bound = subsampling_lower(
                samples, sample_means, n, config.boot_reps, config.alpha, rng
            )
            subsampling_larger_schedule_bound = subsampling_larger_schedule_lower(
                samples, sample_means, n, config.boot_reps, config.alpha, rng
            )
            lf_lower = fs_lower(
                sample_means,
                sample_sds,
                n,
                config.alpha,
                config.mc_reps,
                rng,
                kappa=np.inf,
            )
            fs_lower_bound = fs_lower(
                sample_means,
                sample_sds,
                n,
                config.alpha,
                config.mc_reps,
                rng,
                kappa=kappa,
            )

            for method, lower in (
                ("Oracle", oracle_lower),
                ("Bootstrap", bootstrap_lower_bound),
                ("Numerical bootstrap", numerical_lower_bound),
                ("Subsampling", subsampling_lower_bound),
                ("Subsampling-larger-schedule", subsampling_larger_schedule_bound),
                ("Least-favorable", lf_lower),
                ("Fang-Santos", fs_lower_bound),
            ):
                records.append(
                    {
                        "n": n,
                        "method": method,
                        "lower": lower,
                        "covers": lower <= true_theta,
                    }
                )

    return pd.DataFrame.from_records(records)


def plot_results_one_sided(df, output_dir, prefix=None):
    os.makedirs(output_dir, exist_ok=True)
    summary = (
        df.groupby(["n", "method"], as_index=False)
        .agg(coverage=("covers", "mean"), avg_lower=("lower", "mean"))
    )
    prefix = f"{prefix}-" if prefix else ""

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary["method"].unique():
        subset = summary[summary["method"] == method]
        ax.plot(subset["n"], subset["coverage"], marker="o", label=method)
    ax.axhline(0.975, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Coverage")
    ax.set_title("One-sided Coverage for max(mu)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}coverage.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary["method"].unique():
        subset = summary[summary["method"] == method]
        ax.plot(subset["n"], subset["avg_lower"], marker="o", label=method)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Average lower bound")
    ax.set_title("One-sided Lower Bounds for max(mu)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}lower.png"), dpi=150)
    plt.close(fig)

    summary.to_csv(os.path.join(output_dir, f"{prefix}summary.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate one-sided lower CIs for the maximum of five means."
    )
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--boot-reps", type=int, default=300)
    parser.add_argument("--mc-reps", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="output-one-sided")
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
    df = simulate_config_one_sided(config)
    plot_results_one_sided(df, args.output)


if __name__ == "__main__":
    main()

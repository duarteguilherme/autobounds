"""Bayesian credible intervals for max of Bernoulli means with SBC checks."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = {
    "near-ties": np.array([0.49, 0.495, 0.5, 0.505, 0.51]),
    "boundary": np.array([0.02, 0.05, 0.1, 0.2, 0.3]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian credible intervals for max of Bernoulli means."
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario name (near-ties, boundary) or 'all'.",
    )
    parser.add_argument("--probs", default=None)
    parser.add_argument("--n-values", default="20,50,100,200,500,1000,5000,10000")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--post-reps", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="output-bayesian")
    parser.add_argument(
        "--no-sbc",
        action="store_true",
        help="Disable SBC (prior predictive) diagnostics.",
    )
    return parser.parse_args()


def parse_probs(probs_arg, scenario):
    if probs_arg:
        return np.fromstring(probs_arg, sep=",", dtype=float)
    if scenario in SCENARIOS:
        return SCENARIOS[scenario]
    return SCENARIOS["near-ties"]


def parse_n_values(n_values_arg):
    return [int(value) for value in n_values_arg.split(",")]


def parse_scenarios(scenario_arg):
    if scenario_arg.lower() == "all":
        return list(SCENARIOS.keys())
    return [value.strip() for value in scenario_arg.split(",") if value.strip()]


def posterior_max_draws(samples, post_reps, rng):
    counts = np.array([sample.sum() for sample in samples])
    n = samples[0].shape[0]
    a = counts + 1
    b = (n - counts) + 1
    draws = rng.beta(a, b, size=(post_reps, a.size))
    return draws.max(axis=1)


def credible_interval_from_draws(draws, alpha):
    lower = np.quantile(draws, alpha / 2)
    upper = np.quantile(draws, 1 - alpha / 2)
    return lower, upper


def simulate_fixed(probs, n_values, reps, post_reps, alpha, seed):
    rng = np.random.default_rng(seed)
    records = []
    true_theta = float(probs.max())
    for n in n_values:
        for _ in range(reps):
            samples = [rng.binomial(n=1, p=p, size=n) for p in probs]
            draws = posterior_max_draws(samples, post_reps, rng)
            lower, upper = credible_interval_from_draws(draws, alpha)
            records.append(
                {
                    "n": n,
                    "lower": lower,
                    "upper": upper,
                    "covers": lower <= true_theta <= upper,
                    "length": upper - lower,
                }
            )
    return pd.DataFrame.from_records(records)


def plot_fixed_results(df, output_dir, prefix=None, alpha=0.05):
    os.makedirs(output_dir, exist_ok=True)
    summary = (
        df.groupby(["n"], as_index=False)
        .agg(coverage=("covers", "mean"), avg_length=("length", "mean"))
    )
    prefix = f"{prefix}-" if prefix else ""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n"], summary["coverage"], marker="o", label="Bayesian")
    ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Coverage")
    ax.set_title("Credible Interval Coverage for max(p)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}coverage.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n"], summary["avg_length"], marker="o", label="Bayesian")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Average CI length")
    ax.set_title("Credible Interval Length for max(p)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}length.png"), dpi=150)
    plt.close(fig)

    summary.to_csv(os.path.join(output_dir, f"{prefix}summary.csv"), index=False)


def simulate_sbc(n_values, reps, post_reps, alpha, seed):
    rng = np.random.default_rng(seed)
    records = []
    for n in n_values:
        for _ in range(reps):
            true_ps = rng.beta(1, 1, size=5)
            true_theta = float(true_ps.max())
            samples = [rng.binomial(n=1, p=p, size=n) for p in true_ps]
            draws = posterior_max_draws(samples, post_reps, rng)
            rank = int((draws < true_theta).sum())
            lower, upper = credible_interval_from_draws(draws, alpha)
            records.append(
                {
                    "n": n,
                    "rank": rank,
                    "covers": lower <= true_theta <= upper,
                }
            )
    return pd.DataFrame.from_records(records)


def plot_sbc(df, output_dir, prefix=None, post_reps=2000, alpha=0.05):
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{prefix}-" if prefix else ""
    for n in sorted(df["n"].unique()):
        subset = df[df["n"] == n]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            subset["rank"],
            bins=np.arange(0, post_reps + 2) - 0.5,
            density=True,
            color="#4c78a8",
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Density")
        ax.set_title(f"SBC Rank Histogram (n={n})")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}sbc-ranks-n{n}.png"), dpi=150)
        plt.close(fig)

    coverage = (
        df.groupby(["n"], as_index=False)
        .agg(coverage=("covers", "mean"))
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverage["n"], coverage["coverage"], marker="o", label="SBC")
    ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Coverage")
    ax.set_title("SBC Credible Interval Coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}sbc-coverage.png"), dpi=150)
    plt.close(fig)

    coverage.to_csv(os.path.join(output_dir, f"{prefix}sbc-summary.csv"), index=False)


def main():
    args = parse_args()
    scenarios = parse_scenarios(args.scenario)
    n_values = parse_n_values(args.n_values)

    for scenario in scenarios:
        output_dir = os.path.join(args.output, scenario)
        if not args.no_sbc:
            sbc_df = simulate_sbc(
                n_values,
                args.reps,
                args.post_reps,
                args.alpha,
                args.seed + 1,
            )
            sbc_out = os.path.join(output_dir, "sbc")
            plot_sbc(
                sbc_df,
                sbc_out,
                prefix=scenario,
                post_reps=args.post_reps,
                alpha=args.alpha,
            )


if __name__ == "__main__":
    main()

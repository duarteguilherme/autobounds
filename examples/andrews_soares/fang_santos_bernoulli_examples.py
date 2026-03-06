"""Run one-sided lower CI simulations for Bernoulli Fang-Santos example."""

import argparse
import math

import numpy as np
import pandas as pd

from one_sided_simulation import (
    bootstrap_lower,
    compute_oracle_lower,
    fs_lower,
    numerical_bootstrap_lower,
    plot_results_one_sided,
    subsampling_lower,
    subsampling_larger_schedule_lower,
)
from simulation import SimulationConfig


DEFAULT_PROBS = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
SCENARIOS = {
    "near-ties": np.array([0.49, 0.495, 0.5, 0.505, 0.51]),
    "boundary": np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-sided lower CI simulations for Bernoulli means."
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario name (near-ties, boundary) or 'all'.",
    )
    parser.add_argument("--probs", default=None)
    parser.add_argument("--n-values", default="20,50,100,200,500,1000,5000,10000")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--boot-reps", type=int, default=300)
    parser.add_argument("--post-reps", type=int, default=3000)
    parser.add_argument("--mc-reps", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="output-one-sided")
    return parser.parse_args()


def parse_probs(probs_arg, scenario):
    if probs_arg:
        return np.fromstring(probs_arg, sep=",", dtype=float)
    if scenario in SCENARIOS:
        return SCENARIOS[scenario]
    return DEFAULT_PROBS


def parse_n_values(n_values_arg):
    return [int(value) for value in n_values_arg.split(",")]


def parse_scenarios(scenario_arg):
    if scenario_arg.lower() == "all":
        return list(SCENARIOS.keys())
    return [value.strip() for value in scenario_arg.split(",") if value.strip()]


def bayesian_dirichlet_lower(samples, n, post_reps, alpha, rng):
    successes = np.array([sample.sum() for sample in samples])
    failures = n - successes
    posterior_draws = rng.beta(
        successes + 1, failures + 1, size=(post_reps, successes.size)
    )
    return np.quantile(posterior_draws.max(axis=1), alpha)


def simulate_config_one_sided_bernoulli(config: SimulationConfig, post_reps):
    rng = np.random.default_rng(config.seed)
    records = []
    true_max_index = int(np.argmax(config.true_means))
    true_theta = float(config.true_means.max())

    for n in config.n_values:
        kappa = math.sqrt(math.log(n))
        for _ in range(config.reps):
            samples = [
                rng.binomial(n=1, p=p, size=n).astype(float)
                for p in config.true_means
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
            bayesian_dirichlet_bound = bayesian_dirichlet_lower(
                samples, n, post_reps, config.alpha, rng
            )

            for method, lower in (
                ("Oracle", oracle_lower),
                ("Bootstrap", bootstrap_lower_bound),
                ("Numerical bootstrap", numerical_lower_bound),
                ("Subsampling", subsampling_lower_bound),
                ("Subsampling-larger-schedule", subsampling_larger_schedule_bound),
                ("Least-favorable", lf_lower),
                ("Fang-Santos", fs_lower_bound),
                ("Bayesian-Dirichlet", bayesian_dirichlet_bound),
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


def main():
    args = parse_args()
    scenarios = parse_scenarios(args.scenario)
    n_values = parse_n_values(args.n_values)
    for scenario in scenarios:
        probs = parse_probs(args.probs, scenario)
        config = SimulationConfig(
            true_means=probs,
            n_values=n_values,
            reps=args.reps,
            boot_reps=args.boot_reps,
            mc_reps=args.mc_reps,
            alpha=args.alpha,
            seed=args.seed,
        )
        df = simulate_config_one_sided_bernoulli(config, args.post_reps)
        output_dir = f"{args.output}/{scenario}"
        plot_results_one_sided(df, output_dir, prefix=scenario)


if __name__ == "__main__":
    main()

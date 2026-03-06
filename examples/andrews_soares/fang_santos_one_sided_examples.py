"""Run one-sided lower CI simulations for Fang-Santos examples."""

import argparse

import numpy as np

from one_sided_simulation import plot_results_one_sided, simulate_config_one_sided
from simulation import SimulationConfig


EXAMPLE_CONFIGS = {
    "2.1": np.array([0.0, 0.01, 0.02, 0.03, 0.04]),
    "2.2": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "2.3": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-sided lower CI simulations for Fang-Santos examples."
    )
    parser.add_argument(
        "--examples",
        default="all",
        help="Comma-separated list (e.g. 2.1,2.2) or 'all'.",
    )
    parser.add_argument("--means", default=None)
    parser.add_argument("--n-values", default="50,100,200,500,1000,5000,10000")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--boot-reps", type=int, default=300)
    parser.add_argument("--mc-reps", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="output-one-sided")
    return parser.parse_args()


def parse_means(means_arg, example):
    if means_arg:
        return np.fromstring(means_arg, sep=",", dtype=float)
    return EXAMPLE_CONFIGS[example]


def parse_n_values(n_values_arg):
    return [int(value) for value in n_values_arg.split(",")]


def parse_examples(examples_arg):
    if examples_arg.lower() == "all":
        return list(EXAMPLE_CONFIGS.keys())
    return [value.strip() for value in examples_arg.split(",") if value.strip()]


def main():
    args = parse_args()
    examples = parse_examples(args.examples)
    n_values = parse_n_values(args.n_values)
    for example in examples:
        means = parse_means(args.means, example)
        config = SimulationConfig(
            true_means=means,
            n_values=n_values,
            reps=args.reps,
            boot_reps=args.boot_reps,
            mc_reps=args.mc_reps,
            alpha=args.alpha,
            seed=args.seed,
        )
        df = simulate_config_one_sided(config)
        output_dir = f"{args.output}/fang-santos-example-{example}"
        prefix = f"fang-santos-example-{example}"
        plot_results_one_sided(df, output_dir, prefix=prefix)


if __name__ == "__main__":
    main()

"""Main applications from Duarte et al., JASA 119(547), 1778–1793.

Run from the repository root with:
    python examples/paper_discrete.py --example iv_standard_ate

Uses the maintained package, SCIP, and copies of the original replication data.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from autobounds import Bounder, DAG


DATA = Path(__file__).resolve().with_name('paper_discrete_data')
EXAMPLES = (
    'iv_cautious_ate', 'iv_standard_ate', 'iv_monotone_ate',
    'iv_cautious_late', 'iv_standard_late', 'iv_monotone_late',
    'outcome_selection', 'measurement_error', 'nonresponse', 'joint_missingness',
)


def read_data(filename):
    """Read original rounded probability tables without renormalizing them."""
    return pd.read_csv(DATA / filename)


def build_problem(name):
    """Declare the graph, observed probabilities, assumptions, and estimand."""
    if name not in EXAMPLES:
        raise ValueError(f'Unknown paper example: {name}')
    if name.startswith('iv_'):
        _, assumption, target = name.split('_')
        edges = 'Z -> X, X -> Y, U -> X, U -> Y'
        if assumption == 'cautious':
            edges += ', Z -> Y'
        problem = Bounder(DAG(edges, unob='U'))
        if assumption == 'monotone':
            problem.add_assumption(problem.p('X(Z=0)=1&X(Z=1)=0'))
        problem.load_data(read_data('iv.csv'))
        if target == 'ate':
            problem.set_ate('X', 'Y')
        else:
            compliers = 'X(Z=1)=1&X(Z=0)=0'
            numerator = (problem.p('Y(X=1)=1&' + compliers)
                         - problem.p('Y(X=0)=1&' + compliers))
            problem.set_estimand(numerator, div=problem.p(compliers))
        return problem

    if name == 'outcome_selection':
        problem = Bounder(DAG('Y -> S, X -> Y, U -> X, U -> Y', unob='U'))
        # These are joint probabilities for observed S=1 cells. Their sum is
        # the selection rate, not one: do not condition or renormalize them.
        problem.load_data(read_data('selection_obsqty.csv'))
    elif name == 'measurement_error':
        problem = Bounder(DAG('X -> Y, Y -> S, U -> S, U -> Y', unob='U'))
        problem.load_data(read_data('measurement_error.csv'))
        problem.add_assumption(problem.p('S(Y=0)=1&S(Y=1)=0'))
    elif name == 'nonresponse':
        problem = Bounder(DAG('X -> Y, Y -> R, X -> R'))
        problem.load_data(read_data('manski1_1.csv'), cond=['R'])
        problem.load_data(read_data('manski1_2.csv'))
    else:
        # A and B are the X- and Y-response indicators respectively.
        problem = Bounder(DAG('X -> Y, A -> B, Y -> B'))
        response = read_data('manski2_obs_rx_ry.csv')
        probability_a1 = response.loc[response.A == 1, 'prob'].sum()
        treatment = read_data('manski2_obs_x_rx1.csv')
        # A is independent of X in this graph, so P(X|A=1) identifies P(X).
        treatment['prob'] = treatment['prob'] / probability_a1
        problem.load_data(treatment[['X', 'prob']])
        problem.load_data(response)
        problem.load_data(read_data('manski2_obs_x_y_rx1_ry1.csv'))
        problem.load_data(read_data('manski2_obs_y_ry1.csv'))
    problem.set_ate('X', 'Y')
    return problem


def run_example(name, time_limit=60, gap=1e-5):
    """Solve one declaration, retaining infeasibility and unfinished bounds."""
    program = build_problem(name).write_program()
    program.run_scip(verbose=False, maxtime=time_limit, theta=gap, limits=(-1, 1))
    result = program.last_result
    endpoints = [result.lower, result.upper]
    if all(endpoint.end == 0 for endpoint in endpoints):
        status = 'infeasible'
    elif all(endpoint.primal is not None and endpoint.dual is not None
             and math.isfinite(endpoint.primal) and math.isfinite(endpoint.dual)
             and abs(endpoint.primal - endpoint.dual) <= gap for endpoint in endpoints):
        status = 'converged'
    else:
        status = 'unfinished'
    def endpoint_dict(endpoint):
        return {key: value if math.isfinite(value) else None
                for key, value in endpoint.to_dict().items()}

    return {'example': name, 'status': status,
            'lower': endpoint_dict(result.lower), 'upper': endpoint_dict(result.upper)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=(*EXAMPLES, 'all'), default='iv_standard_ate')
    parser.add_argument('--time-limit', type=float, default=60,
                        help='Maximum solve time per example, in seconds.')
    parser.add_argument('--gap', type=float, default=1e-5)
    parser.add_argument('--output', type=Path, help='Optional JSON results file.')
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.time_limit, args.gap)):
        parser.error('--time-limit and --gap must be finite and positive')
    names = EXAMPLES if args.example == 'all' else (args.example,)
    results = []
    for name in names:
        result = run_example(name, args.time_limit, args.gap)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()

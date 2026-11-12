"""Reproduce the paper's main numerical applications with the installed API."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.paper_discrete import EXAMPLES, build_problem, read_data, run_example


@pytest.mark.parametrize('name,expected', [
    ('iv_cautious_ate', (-.627596, .372404)),
    ('iv_standard_ate', (-.5502, -.1460)),
    ('iv_cautious_late', (-1, 1)),
    ('iv_standard_late', (-1, 1)),
    ('outcome_selection', (-.50, .64)),
    ('measurement_error', (-.62, 1)),
    ('nonresponse', (-.25, .75)),
])
def test_main_paper_bounds_match_published_results(name, expected):
    result = run_example(name, time_limit=30)
    assert result['status'] == 'converged', result
    for endpoint, value in zip(('lower', 'upper'), expected):
        assert result[endpoint]['primal'] == pytest.approx(value, abs=.005)
        assert result[endpoint]['dual'] == pytest.approx(value, abs=.005)
        assert abs(result[endpoint]['primal'] - result[endpoint]['dual']) <= 1e-5


@pytest.mark.parametrize('name', ['iv_monotone_ate', 'iv_monotone_late'])
def test_false_monotonicity_is_reported_as_infeasible(name):
    result = run_example(name, time_limit=10)
    assert result['status'] == 'infeasible'
    assert result['lower']['end'] == result['upper']['end'] == 0
    assert 'primal' not in result['lower']


def shadow_variable_ate():
    """Independent identification calculation using the two treatment levels.

    For A=1 complete cases, f_xy/P(X=x) = P(Y=y|X=x)*P(A=1)*s_y.
    Solving the two row-normalization equations identifies the inverse
    selection factors and hence the two outcome means.
    """
    response = read_data('manski2_obs_rx_ry.csv')
    treatment = read_data('manski2_obs_x_rx1.csv').set_index('X')['prob']
    treatment = treatment / response.loc[response.A == 1, 'prob'].sum()
    complete = read_data('manski2_obs_x_y_rx1_ry1.csv')
    matrix = complete.pivot(index='X', columns='Y', values='prob').div(treatment, axis=0)
    inverse_selection = np.linalg.solve(matrix.to_numpy(), np.ones(2))
    means = matrix[1].to_numpy() * inverse_selection[1]
    return means[1] - means[0]


def test_joint_missingness_brackets_independent_identification_result():
    expected = shadow_variable_ate()
    assert expected == pytest.approx(-.25, abs=.005)
    result = run_example('joint_missingness', time_limit=10)
    assert result['status'] in ('converged', 'unfinished')
    # This nonlinear example may time out in CI. Check valid outer bounds and
    # near-identifying feasible witnesses, without calling unfinished bounds sharp.
    assert result['lower']['dual'] <= expected + 2e-4
    assert result['upper']['dual'] >= expected - 2e-4
    assert result['lower']['primal'] == pytest.approx(expected, abs=2e-4)
    assert result['upper']['primal'] == pytest.approx(expected, abs=2e-4)


def test_selected_joint_probabilities_are_not_renormalized():
    assert read_data('selection_obsqty.csv')['prob'].sum() == pytest.approx(.309208)
    assert read_data('manski1_1.csv')['prob'].sum() == pytest.approx(1)


@pytest.mark.parametrize('name', EXAMPLES)
def test_all_paper_declarations_compile(name):
    model = build_problem(name).compile()
    assert model.constraints and model.variables


def test_cli_works_outside_checkout_and_writes_json(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'examples' / 'paper_discrete.py'
    output = tmp_path / 'result.json'
    result = subprocess.run([
        sys.executable, str(script), '--example', 'iv_standard_ate',
        '--output', str(output), '--time-limit', '10',
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    records = json.loads(output.read_text())
    assert records[0]['status'] == 'converged'
    assert json.loads(result.stdout) == records[0]

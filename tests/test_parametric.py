"""Automatic reductions start with graph declarations, never an IV formula."""
import random

import pytest

pytest.importorskip('cdd.gmp')

from autobounds import DAG, SymbolicProblem
from autobounds.backends.scip import ScipBackend
from tests.homogeneous_iv_oracle import homogeneous_iv_bounds
from tests.test_homogeneous_iv import full_model


def declaration(*, names=('Z', 'D', 'Y'), exclusion=False, homogeneous=True, mode=None):
    z, d, y = names
    edges = f'{z} -> {d}, {d} -> {y}, U -> {d}, U -> {y}'
    if not exclusion:
        edges += f', {z} -> {y}'
    problem = SymbolicProblem(DAG(edges, unob='U'), parametric=mode)
    problem.set_p_to_zero(f'{d}({z}=0)=1&{d}({z}=1)=0')
    problem.load_data(f'{d},{y}', cond=z)
    problem.load_data(z)
    problem.set_ate(d, y)
    if homogeneous:
        problem.add_homogeneity(f'{y}({d}=1)=1', f'{y}({d}=0)=1', strata=[
            f'{d}({z}=0)=0&{d}({z}=1)=0',
            f'{d}({z}=0)=0&{d}({z}=1)=1',
            f'{d}({z}=0)=1&{d}({z}=1)=1'])
    return problem


def cells(values, names=('Z', 'D', 'Y')):
    z, d, y = names
    a, n, w, g0, g1, b0, b1 = [values[k] for k in ('a', 'n', 'w', 'g0', 'g1', 'b0', 'b1')]
    table = [[1-a-b0, b0, a-g0, g0], [n-g1, g1, 1-n-b1, b1]]
    return {**{f'P({d}={i}&{y}={j}|{z}={k})': table[k][2*i+j]
               for k in range(2) for i in range(2) for j in range(2)},
            f'P({z}=0)': 1-w, f'P({z}=1)': w}


@pytest.fixture(scope='module')
def automatic():
    reduction = declaration().write_program().linearize()
    return reduction, reduction.solve()


def assert_endpoints(result, expected):
    for endpoint, value in zip((result.lower, result.upper), expected):
        assert endpoint.primal == pytest.approx(value, abs=2e-6)
        assert endpoint.dual == pytest.approx(value, abs=2e-6)


@pytest.mark.parametrize('seed', range(12))
def test_automatic_reduction_matches_full_model_and_independent_oracle(automatic, seed):
    reduction, bounds = automatic
    rng = random.Random(seed)
    a = rng.randrange(1, 7)/8
    n = rng.randrange(1, 8-int(a*8))/8
    values = dict(a=a, n=n, w=rng.randrange(1, 8)/8,
                  g0=a*rng.randrange(9)/8, g1=n*rng.randrange(9)/8,
                  b0=(1-a)*rng.randrange(9)/8, b1=(1-n)*rng.randrange(9)/8)
    expected = homogeneous_iv_bounds().evaluate(values)
    data = cells(values)
    assert bounds.evaluate(data) == pytest.approx(expected)
    original = full_model(values)
    reduced = reduction.model.bind(data)
    assert original.is_linear and reduced.is_linear
    assert_endpoints(ScipBackend().solve(original, verbose=False), expected)
    assert_endpoints(ScipBackend().solve(reduced, verbose=False), expected)


@pytest.mark.parametrize('a,n', [(0, 0), (0, .5), (.5, 0), (.5, .5), (1, 0), (0, 1)])
def test_automatic_projection_keeps_zero_mass_strata(automatic, a, n):
    values = dict(a=a, n=n, w=.25, g0=a*.75, g1=n*.25,
                  b0=(1-a)*.25, b1=(1-n)*.75)
    assert automatic[1].evaluate(cells(values)) == pytest.approx(homogeneous_iv_bounds().evaluate(values))


def test_reduction_is_derived_and_inspectable(automatic):
    reduction, bounds = automatic
    assert reduction.report['response_types'] == 48
    assert reduction.report['removed_response_types'] == 16
    assert reduction.report['response_features'] == 13
    assert reduction.report['hull_constraints'] == 33
    assert reduction.report['nuisance_variables'] == 3
    assert reduction.report['is_parametric_linear']
    assert len(bounds.report['projection_steps']) == 3


def test_renaming_graph_variables_does_not_change_solution():
    names = ('Instrument', 'Treatment', 'Outcome')
    bounds = declaration(names=names).solve()
    values = dict(a=.25, n=.25, w=.5, g0=.25, g1=.25, b0=0, b1=.75)
    assert bounds.evaluate(cells(values, names)) == (0, .5)


def test_changing_graph_to_exclusion_identifies_wald_effect():
    bounds = declaration(exclusion=True).solve()
    values = dict(a=.25, n=.25, w=.25, g0=.1875, g1=.0625, b0=.1875, b1=.5625)
    assert bounds.evaluate(cells(values)) == pytest.approx((.5, .5))


def test_removing_homogeneity_changes_bounds():
    bounds = declaration(homogeneous=False).solve()
    values = dict(a=.25, n=.25, w=.5, g0=.25, g1=.25, b0=0, b1=.75)
    expected = (-.125, .875)
    assert bounds.evaluate(cells(values)) == expected
    assert_endpoints(ScipBackend().solve(full_model(values, homogeneous=False), verbose=False), expected)


def test_missing_invalid_and_incompatible_data_are_rejected(automatic):
    bounds = automatic[1]
    values = dict(a=.25, n=.25, w=.5, g0=.25, g1=.25, b0=0, b1=.75)
    data = cells(values)
    with pytest.raises(ValueError, match='Missing'):
        bounds.evaluate({})
    for key, value in [('P(Z=0)', 0), ('P(D=0&Y=0|Z=0)', -1), ('P(Z=1)', .25)]:
        with pytest.raises(ValueError):
            bounds.evaluate(dict(data, **{key: value}))
    values.update(a=.75, n=.75, g0=0, g1=0, b0=0, b1=0)
    with pytest.raises(ValueError, match='incompatible'):
        bounds.evaluate(cells(values))


def test_invalid_partition_is_not_silently_accepted():
    problem = declaration(homogeneous=False)
    problem.add_homogeneity('Y(D=1)=1', 'Y(D=0)=1', strata=['D(Z=0)=0', 'D(Z=1)=1'])
    with pytest.raises(ValueError, match='partition'):
        problem.solve()


def test_projection_budget_does_not_return_relaxed_bounds(automatic):
    with pytest.raises(ValueError, match='max_rows'):
        automatic[0].solve(max_rows=1)


def test_general_confounded_graph_without_instrument():
    problem = SymbolicProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'), parametric=True)
    problem.load_data('X,Y')
    problem.set_ate('X', 'Y')
    bounds = problem.solve()
    assert bounds.evaluate({'P(X=0&Y=0)': .125, 'P(X=0&Y=1)': .125,
                            'P(X=1&Y=0)': .25, 'P(X=1&Y=1)': .5}) == (-.375, .625)


def test_unresolved_nonlinearity_is_rejected():
    problem = SymbolicProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'), parametric=True)
    problem.load_data('X,Y')
    event = problem.p('Y(X=1)=1')
    problem.set_estimand(event*event)
    with pytest.raises(ValueError, match='nonlinear response products'):
        problem.solve()


def test_feature_hull_contains_every_allowed_response_vertex(automatic):
    reduction = automatic[0]
    response_symbols = set().union(*(e.free_symbols for _, e in reduction.feature_map))
    assert len(response_symbols) == 48
    for response in response_symbols:
        vertex = {q: int(q == response) for q in response_symbols}
        point = {x: e.xreplace(vertex) for x, e in reduction.feature_map}
        for e, relation in reduction.hull_rows:
            value = e.xreplace(point)
            assert value == 0 if relation == '==' else value >= 0


def test_symbolic_envelopes_evaluate_at_zero_stratum_boundary(automatic):
    bounds = automatic[1]
    values = cells(dict(a=0, n=0, w=.5, g0=0, g1=0, b0=.25, b1=.75))
    sub = {p: values[str(p)] for p in bounds.symbols}
    expected = bounds.evaluate(values)
    assert (float(bounds.lower.xreplace(sub)), float(bounds.upper.xreplace(sub))) == expected


def test_candidates_are_separated_and_preserve_boundary_evaluation(automatic):
    bounds = automatic[1]
    assert len(bounds.lower_candidates) == len(bounds.upper_candidates) == 37
    for a, n in [(.25, .25), (0, 0), (.5, .5), (1, 0), (0, 1)]:
        data = cells(dict(a=a, n=n, w=.25, g0=a*.75, g1=n*.25,
                          b0=(1-a)*.25, b1=(1-n)*.75))
        sub = {p: data[str(p)] for p in bounds.symbols}
        lower = [float(c.expression.xreplace(sub)) for c in bounds.lower_candidates
                 if c.active_when.xreplace(sub)]
        upper = [float(c.expression.xreplace(sub)) for c in bounds.upper_candidates
                 if c.active_when.xreplace(sub)]
        assert (max(lower), min(upper)) == pytest.approx(bounds.evaluate(data))
    text = bounds.format_candidates()
    assert 'Lower candidates (37)' in text and 'Upper candidates (37)' in text
    assert 'L1 =' in text and 'U1 =' in text and 'A = ' not in text
    assert 'Compatibility conditions:' not in text
    assert 'Compatibility conditions:' in bounds.format_candidates(include_compatibility=True)
    assert r'\frac' in bounds.format_candidates(style='latex')


def test_candidate_display_keeps_unknown_signs_and_zero_coefficient_conditions():
    import sympy as sp
    from autobounds.parametric import ParametricBounds
    p, t = sp.symbols('p t', real=True)
    bounds = ParametricBounds((p,), t, (), (p*t-1,), (), {})
    assert bounds.lower_candidates[0].active_when == (p > 0)
    assert bounds.upper_candidates[0].active_when == (p < 0)
    assert all(condition.subs(p, 0) == sp.false for condition in bounds.compatibility_conditions)
    with pytest.raises(ValueError, match='style'):
        bounds.format_candidates(style='html')

"""Generic symbolic reduction and display regressions."""
from itertools import product

import pandas as pd
import pytest

pytest.importorskip('cdd.gmp')

from autobounds import Bounder, DAG, SymbolicProblem
from autobounds.backends.scip import ScipBackend


EDGES = 'X -> Y, G -> Y, U -> X, U -> Y'


def declaration():
    problem = SymbolicProblem(DAG(EDGES, unob='U'))
    problem.load_data('X,Y', cond='G')
    problem.load_data('G')
    problem.set_ate('X', 'Y')
    problem.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    return problem


def cells(mass, weight):
    return {**{f'P(X={x}&Y={y}|G={g})':
               ([1-mass, mass][x] * ([.75, .25] if g == 0 else [.25, .75])[y])
               for g, x, y in product(range(2), repeat=3)},
            'P(G=0)': 1-weight, 'P(G=1)': weight}


def numeric_model(mass, weight):
    problem = Bounder(DAG(EDGES, unob='U'))
    data = cells(mass, weight)
    problem.load_data(pd.DataFrame([
        (g, x, y, data[f'P(X={x}&Y={y}|G={g})']*[1-weight, weight][g])
        for g, x, y in product(range(2), repeat=3)],
        columns=['G', 'X', 'Y', 'prob']))
    effect = problem.p('Y(X=1)=1')-problem.p('Y(X=0)=1')
    for x, probability in [(0, 1-mass), (1, mass)]:
        numerator = (problem.p(f'Y(X=1)=1&X={x}')
                     - problem.p(f'Y(X=0)=1&X={x}'))
        problem.add_constraint(numerator-probability*effect)
    problem.set_estimand(effect)
    return problem.compile()


@pytest.fixture(scope='module')
def automatic():
    reduction = declaration().write_program().linearize()
    return reduction, reduction.solve()


@pytest.mark.parametrize('mass,weight', [(0, .25), (.25, .5), (.5, .75), (1, .5)])
def test_reduced_and_full_models_agree_including_empty_groups(automatic, mass, weight):
    reduction, bounds = automatic
    data = cells(mass, weight)
    expected = bounds.evaluate(data)
    for model in [numeric_model(mass, weight), reduction.model.bind(data)]:
        assert model.is_linear
        result = ScipBackend().solve(model, verbose=False)
        for endpoint, value in zip((result.lower, result.upper), expected):
            assert endpoint.primal == pytest.approx(value, abs=2e-6)
            assert endpoint.dual == pytest.approx(value, abs=2e-6)
    sub = {p: data[str(p)] for p in bounds.symbols}
    lower = [float(c.expression.xreplace(sub)) for c in bounds.lower_candidates
             if c.active_when.xreplace(sub)]
    upper = [float(c.expression.xreplace(sub)) for c in bounds.upper_candidates
             if c.active_when.xreplace(sub)]
    assert (max(lower), min(upper)) == pytest.approx(expected)


def test_feature_hull_contains_every_response_vertex(automatic):
    reduction = automatic[0]
    symbols = set().union(*(e.free_symbols for _, e in reduction.feature_map))
    for response in symbols:
        vertex = {q: int(q == response) for q in symbols}
        point = {x: e.xreplace(vertex) for x, e in reduction.feature_map}
        for e, relation in reduction.hull_rows:
            value = e.xreplace(point)
            assert value == 0 if relation == '==' else value >= 0


def test_candidates_and_compatibility_are_separate(automatic):
    bounds = automatic[1]
    text = bounds.format_candidates()
    assert 'L1 =' in text and 'U1 =' in text
    assert 'Compatibility conditions:' not in text
    assert 'Compatibility conditions:' in bounds.format_candidates(include_compatibility=True)
    assert r'\frac' in bounds.format_candidates(style='latex')
    with pytest.raises(ValueError, match='Missing'):
        bounds.evaluate({})
    invalid = cells(.5, .5)
    invalid['P(G=0)'] = 0
    with pytest.raises(ValueError, match='strictly positive'):
        bounds.evaluate(invalid)


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

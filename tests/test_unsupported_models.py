"""Capability limits are explicit errors, not infeasible or approximate bounds."""
import pytest
import sympy as sp

from autobounds import DAG, SymbolicProblem, UnsupportedModelError
from autobounds.parametric import ParametricReduction


def problem():
    result = SymbolicProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'))
    result.load_data('X,Y')
    result.set_ate('X', 'Y')
    return result


def test_overlapping_components_have_a_specific_support_error():
    with pytest.raises(UnsupportedModelError, match='common latent parent'):
        SymbolicProblem(DAG('U -> X, U -> Y, V -> Y, V -> Z', unob='U,V'))


def test_conditioning_on_nonroot_is_not_treated_as_intervention():
    with pytest.raises(UnsupportedModelError, match='independent observed roots'):
        problem().load_data('Y', cond='X')


def test_unresolved_response_product_has_a_specific_support_error():
    pytest.importorskip('cdd.gmp')
    p = problem()
    event = p.p('Y(X=1)=1')
    p.set_estimand(event*event)
    with pytest.raises(UnsupportedModelError, match='Unresolved nonlinear response products'):
        p.solve()


def test_different_homogeneity_effect_is_explicitly_unsupported():
    pytest.importorskip('cdd.gmp')
    p = problem()
    p.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    p.set_estimand(p.p('Y(X=1)=1'))
    with pytest.raises(UnsupportedModelError, match='match the declared estimand'):
        p.solve()


def test_unknown_nuisance_sign_requires_case_splitting():
    x, t, p = sp.symbols('x t p', real=True)
    reduction = ParametricReduction(None, (x,), (p,), t, (), (p*x-t, 1-x), (), {})
    with pytest.raises(UnsupportedModelError, match='parameter case splitting'):
        reduction.solve()


def test_budget_limit_is_not_misreported_as_an_unsupported_model():
    x, t = sp.symbols('x t', real=True)
    reduction = ParametricReduction(None, (x,), (), t, (), (x-t, 1-x, 2-x), (), {})
    with pytest.raises(ValueError, match='max_rows=1') as caught:
        reduction.solve(max_rows=1)
    assert not isinstance(caught.value, UnsupportedModelError)
    assert 'generate 2 rows' in str(caught.value)
    assert reduction.solve(max_rows=2).evaluate({}) == (-float('inf'), 1)


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_budget_has_an_input_error(limit):
    t = sp.Symbol('t', real=True)
    reduction = ParametricReduction(None, (), (), t, (), (t,), (), {})
    with pytest.raises(ValueError, match='positive integer'):
        reduction.solve(max_rows=limit)


def test_invalid_partition_is_an_input_error_not_a_solver_bug():
    pytest.importorskip('cdd.gmp')
    p = problem()
    p.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=0'])
    with pytest.raises(ValueError, match='partition') as caught:
        p.solve()
    assert not isinstance(caught.value, UnsupportedModelError)


def test_missing_dependency_is_not_swallowed_or_misreported(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_cdd(name, *args, **kwargs):
        if name == 'cdd.gmp':
            raise ModuleNotFoundError('test: missing cdd.gmp')
        return original(name, *args, **kwargs)
    p = problem()
    p.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    monkeypatch.setattr(builtins, '__import__', without_cdd)
    with pytest.raises(ModuleNotFoundError, match='missing cdd.gmp'):
        p.solve()


def test_fully_identified_components_report_the_actual_capability_limit():
    pytest.importorskip('cdd.gmp')
    from autobounds.parametric import ParametricResponseProgram
    q0, q1 = sp.symbols('q0 q1')
    program = ParametricResponseProgram(((q0, q1),), {'observed': q1}, {}, (), q1, (), ())
    with pytest.raises(UnsupportedModelError, match='fully identified model'):
        program.linearize()

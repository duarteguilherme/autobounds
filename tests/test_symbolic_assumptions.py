"""One public assumption API separates exclusions from population restrictions."""
from fractions import Fraction

import pytest

from autobounds import DAG, SymbolicProblem
from autobounds.parametric import ParametricResponseProgram


def problem():
    return SymbolicProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'))


def track_zeros(p, monkeypatch):
    calls = []
    original = p.set_p_to_zero
    def record(event):
        calls.append(event)
        return original(event)
    monkeypatch.setattr(p, 'set_p_to_zero', record)
    return calls


def test_impossible_event_uses_zero_elimination_internally(monkeypatch):
    p = problem()
    calls = track_zeros(p, monkeypatch)
    p.add_assumption('Y(X=0)=1&Y(X=1)=0')
    p.set_ate('X', 'Y')
    assert len(calls) == 1 and not p._assumptions
    assert len(p.write_program().response_types) == 6


@pytest.mark.parametrize('coefficient,relation', [(2, '=='), (2, '<='), (-2, '=='), (-2, '>=')])
def test_signed_zero_constraints_remove_only_proved_types(monkeypatch, coefficient, relation):
    p = problem()
    calls = track_zeros(p, monkeypatch)
    event = p.p('Y(X=0)=1&Y(X=1)=0')
    p.add_assumption(coefficient*event, relation)
    p.set_ate('X', 'Y')
    assert len(calls) == 1
    assert len(p.write_program().response_types) == 6


def test_probability_one_excludes_the_complement(monkeypatch):
    p = problem()
    calls = track_zeros(p, monkeypatch)
    p.add_assumption(p.p('X=1'), '==', 1)
    p.set_ate('X', 'Y')
    assert len(calls) == 1
    assert len(p.write_program().response_types) == 4


def test_equal_averages_do_not_remove_individual_types(monkeypatch):
    pytest.importorskip('cdd.gmp')
    p = problem()
    calls = track_zeros(p, monkeypatch)
    p.add_assumption(p.p('Y(X=1)=1'), '==', p.p('Y(X=0)=1'))
    p.set_ate('X', 'Y')
    assert calls == [] and len(p._assumptions) == 1
    assert isinstance(p.write_program(), ParametricResponseProgram)
    assert p.solve().evaluate({}) == (0, 0)
    # Equal averages still permit both beneficial and harmful individual types.
    p.set_estimand(p.p('Y(X=1)=1&Y(X=0)=0'))
    assert p.solve().evaluate({}) == (0, .5)


def test_population_probability_value_is_not_an_exclusion(monkeypatch):
    pytest.importorskip('cdd.gmp')
    p = problem()
    calls = track_zeros(p, monkeypatch)
    p.add_assumption(p.p('Y(X=1)=1'), '==', Fraction(1, 4))
    p.set_estimand(p.p('Y(X=1)=1'))
    assert calls == []
    assert p.solve().evaluate({}) == (.25, .25)


def test_population_inequality_is_retained(monkeypatch):
    pytest.importorskip('cdd.gmp')
    p = problem()
    calls = track_zeros(p, monkeypatch)
    effect = p.p('Y(X=1)=1')-p.p('Y(X=0)=1')
    p.add_assumption(effect, '>=', Fraction(1, 4))
    p.set_estimand(effect)
    assert calls == []
    assert p.solve().evaluate({}) == (.25, 1)


def test_product_zero_does_not_force_both_factors_to_zero(monkeypatch):
    p = problem()
    calls = track_zeros(p, monkeypatch)
    p.add_assumption(p.p('Y(X=1)=1')*p.p('Y(X=0)=1'))
    assert calls == [] and len(p._assumptions) == 1


def test_homogeneity_goes_through_population_path(monkeypatch):
    p = problem()
    calls = track_zeros(p, monkeypatch)
    p.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    assert calls == [] and len(p._homogeneity) == 1


def test_contradictory_assumption_does_not_mutate_declarations():
    p = problem()
    with pytest.raises(ValueError, match='contradictory'):
        p.add_assumption(1)
    assert not p._zero_events and not p._assumptions


def test_invalid_assumption_relation_is_rejected_without_mutation():
    p = problem()
    with pytest.raises(ValueError, match='relation must be'):
        p.add_assumption(p.p('X=0'), '!=', 0)
    assert not p._assumptions and not p._zero_events


def test_population_assumption_can_identify_an_independent_root():
    pytest.importorskip('cdd.gmp')
    from tests.test_parametric_general import EDGES, cells, numeric_model
    from autobounds.backends.scip import ScipBackend
    p = SymbolicProblem(DAG(EDGES, unob='U'))
    p.load_data('X,Y', cond='G')
    p.add_assumption(p.p('G=1'), '==', Fraction(1, 2))
    p.set_ate('X', 'Y')
    p.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    bounds = p.solve()
    data = {k: v for k, v in cells(.5, .5).items() if '|G=' in k}
    expected = bounds.evaluate(data)
    result = ScipBackend().solve(numeric_model(.5, .5), verbose=False)
    for endpoint, value in zip((result.lower, result.upper), expected):
        assert endpoint.primal == pytest.approx(value)
        assert endpoint.dual == pytest.approx(value)


def test_exclusion_conflicting_with_conditioning_support_is_rejected():
    pytest.importorskip('cdd.gmp')
    p = SymbolicProblem(DAG('X -> Y, G -> Y, U -> X, U -> Y', unob='U'))
    p.load_data('X,Y', cond='G')
    p.add_assumption('G=1')
    p.set_ate('X', 'Y')
    with pytest.raises(ValueError, match='strictly positive root probabilities'):
        p.solve()


def test_explicit_probability_equality_uses_internal_zero_elimination(monkeypatch):
    p = SymbolicProblem(DAG('Z -> D, D -> Y, Z -> Y, U -> D, U -> Y', unob='U'))
    calls = track_zeros(p, monkeypatch)
    P = p.p
    p.add_assumption(P('D(Z=0)=1'), '==', 0)
    p.set_estimand(P('D(Z=0)=1'))
    program = p.write_program()
    assert len(calls) == 1
    assert len(program.response_types) == 32
    assert set(program.target) == {0}


def test_population_upper_inequality_is_not_an_individual_exclusion(monkeypatch):
    pytest.importorskip('cdd.gmp')
    p = problem()
    calls = track_zeros(p, monkeypatch)
    effect = p.p('Y(X=1)=1')-p.p('Y(X=0)=1')
    p.add_assumption(effect, '<=', -Fraction(1, 4))
    p.set_estimand(effect)
    assert not calls
    assert p.solve().evaluate({}) == (-1, -.25)

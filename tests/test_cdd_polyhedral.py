"""Shared reductions feed exact polyhedral projection, with symbolic data."""
from fractions import Fraction

import pytest
import sympy as sp

pytest.importorskip('cdd.gmp')

from autobounds import LinearResponseModel, Variable
from autobounds.backends.cdd_polyhedral import CddPolyhedralBackend
from autobounds.backends.scip import ScipBackend
from tests.test_reductions import model, poly


def test_symbolic_fixed_sum_reduction_then_cdd():
    original = model([Variable('a'), Variable('b'), Variable('x'),
                      Variable('p', None, None, 'parameter')], [
        (poly((1, 'a'), (1, 'b'), (-1,)), '=='),
        (poly((1, 'a', 'x'), (1, 'b', 'x'), (-1, 'p')), '>='),
    ], poly((1, 'x')))
    reduced = original.linearize()
    assert reduced.model.is_linear
    bounds = reduced.solve('cdd')
    p = bounds.symbols[0]
    assert bounds.lower == sp.Max(0, p)
    assert bounds.upper == 1
    assert bounds.lower.subs(p, 2) > bounds.upper
    for value in [Fraction(0), Fraction(1, 3), Fraction(1)]:
        expected = (float(value), 1)
        assert bounds.evaluate({'p': value}) == pytest.approx(expected)
        numeric = ScipBackend().solve(original.bind({'p': value}), verbose=False)
        assert (numeric.lower.dual, numeric.upper.dual) == pytest.approx(expected)


def test_general_affine_constraints_domains_and_objective_constant():
    original = model([Variable('x', -2, 3), Variable('y', 0, 4),
                      Variable('p', -1, 1, 'parameter', 'observed')], [
        (poly((1, 'x'), (1, 'y'), (-1, 'p')), '=='),
        (poly((2, 'y'), (-3,)), '<='),
    ], poly((2, 'x'), (1,)))
    bounds = original.linearize().solve('cdd')
    for value in [Fraction(-1), Fraction(0), Fraction(1)]:
        numeric = ScipBackend().solve(original.bind({'p': value}), verbose=False)
        assert bounds.evaluate({'observed': value}) == pytest.approx(
            (numeric.lower.dual, numeric.upper.dual))


def test_existing_response_model_uses_shared_pipeline():
    declaration = LinearResponseModel([[1, 0, 1]], [0, 0, 1], labels=['observed'])
    baseline = declaration.solve()
    reduced = declaration.solve(linearize=True)
    assert reduced.lower == baseline.lower
    assert reduced.upper == baseline.upper


def test_data_dependent_decision_coefficients_are_not_silently_projected():
    original = model([Variable('x'), Variable('p', 1, 2, 'parameter')], [
        (poly((1, 'p', 'x'), (-1,)), '=='),
    ], poly((1, 'x')))
    reduced = original.linearize()
    assert not reduced.model.is_linear
    with pytest.raises(ValueError, match='jointly affine'):
        reduced.solve('cdd')
    assert original.bind({'p': 2}).linearize().solve('cdd').evaluate({}) == (.5, .5)


def test_polyhedral_unbounded_infeasible_and_constant_objectives():
    free = model([Variable('x', None, None)], [], poly((1, 'x')))
    assert CddPolyhedralBackend().solve(free).evaluate({}) == (-float('inf'), float('inf'))
    constant = model([], [], poly((Fraction(2, 3),)))
    assert CddPolyhedralBackend().solve(constant).evaluate({}) == (2/3, 2/3)
    impossible = model([Variable('x')], [(poly((1, 'x'), (-2,)), '==')], poly((1, 'x')))
    with pytest.raises(ValueError, match='Infeasible'):
        CddPolyhedralBackend().solve(impossible)


def test_parameter_equalities_survive_projection():
    original = model([Variable('x'), Variable('p', None, None, 'parameter'),
                      Variable('r', None, None, 'parameter')], [
        (poly((1, 'x'), (-1, 'p')), '=='),
        (poly((1, 'x'), (-1, 'r')), '=='),
    ], poly((1, 'x')))
    bounds = original.linearize().solve('cdd')
    p, r = bounds.symbols
    assert any(c.subs({p: 0, r: 1}) == sp.false for c in bounds.compatibility_equalities)

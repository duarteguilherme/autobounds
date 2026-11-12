"""Exact reductions preserve domains, objectives, and reconstruct witnesses."""
from fractions import Fraction as F
from itertools import product

import pytest

from autobounds import Constraint, MathematicalModel, Polynomial, Term, Variable
from autobounds.backends.scip import ScipBackend
from autobounds.reductions import reduce_model


def poly(*terms):
    return Polynomial(tuple(Term(c, tuple(names)) for c, *names in terms))


def model(variables, rows, objective):
    return MathematicalModel(tuple(variables), tuple(Constraint(p, relation)
                             for p, relation in rows), objective)


def check_witness(original, values):
    for v in original.variables:
        assert v.lower is None or values[v.name] >= float(v.lower)-2e-6
        assert v.upper is None or values[v.name] <= float(v.upper)+2e-6
    for row in original.constraints:
        residual = float(row.expression.evaluate(values))
        assert {'==': abs(residual) <= 3e-6, '<=': residual <= 3e-6,
                '>=': residual >= -3e-6}[row.relation]


def check_bounds(original, reduced, expected):
    for result in (ScipBackend().solve(original, verbose=False, maxtime=10),
                   reduced.solve(verbose=False, maxtime=10, return_dgps=True)):
        for endpoint, value in zip((result.lower, result.upper), expected):
            assert endpoint.primal == pytest.approx(value, abs=2e-6)
            assert endpoint.dual == pytest.approx(value, abs=2e-6)
    for key in ('lower', 'upper'):
        values = result.dgps[key]['values']
        check_witness(original, values)
        assert float(original.objective.evaluate(values)) == pytest.approx(
            getattr(result, key).primal, abs=2e-6)


def test_linear_system_identifies_constants_and_exact_repeating_fractions():
    original = model([Variable('a'), Variable('b'), Variable('x')], [
        (poly((1, 'a'), (1, 'b'), (-1,)), '=='),
        (poly((2, 'a'), (-1, 'b')), '=='),
        (poly((1, 'a', 'x'), (1, 'b', 'x'), (-F(1, 2),)), '<='),
    ], poly((1, 'x')))
    reduced = reduce_model(original, fractional=False)
    assert reduced.model.is_linear
    assert any(r.name == 'a' and r.numerator.constant == F(1, 3)
               for r in reduced.reconstruction)
    check_bounds(original, reduced, (0, .5))


def test_fixed_sum_factor_is_linearized_without_identifying_its_parts():
    original = model([Variable('a'), Variable('b'), Variable('x')], [
        (poly((1, 'a'), (1, 'b'), (-F(2, 5),)), '=='),
        (poly((1, 'a', 'x'), (1, 'b', 'x'), (-F(1, 5),)), '<='),
    ], poly((1, 'x')))
    reduced = reduce_model(original, fractional=False)
    assert reduced.model.is_linear
    check_bounds(original, reduced, (0, .5))


def test_affine_substitution_preserves_removed_variable_bounds():
    original = model([Variable('x', 0, F(1, 5)), Variable('y')], [
        (poly((1, 'x'), (1, 'y'), (-1,)), '=='),
    ], poly((1, 'y')))
    reduced = reduce_model(original)
    assert len(reduced.model.variables) < 2
    check_bounds(original, reduced, (.8, 1))


@pytest.mark.parametrize('lower,upper,relation,expected', [
    (1, 2, '<=', (-1, 0)),
    (-2, -1, '<=', (0, 1)),
    (-2, -1, '>=', (-1, 0)),
    (1, 2, '==', (0, 0)),
])
def test_nonzero_factor_cancellation_tracks_inequality_sign(lower, upper, relation, expected):
    original = model([Variable('a', lower, upper), Variable('x', -1, 1)], [
        (poly((1, 'a', 'x')), relation),
    ], poly((1, 'x')))
    reduced = reduce_model(original, fractional=False)
    assert reduced.model.is_linear
    assert any(s[0] in ('cancel_nonzero_factors', 'zero_variables') for s in reduced.steps)
    check_bounds(original, reduced, expected)


def test_zero_factor_branch_is_not_discarded():
    original = model([Variable('a'), Variable('x')], [
        (poly((1, 'a', 'x')), '=='),
    ], poly((1, 'x')))
    reduced = reduce_model(original)
    assert not reduced.model.is_linear
    check_bounds(original, reduced, (0, 1))


def test_unknown_sign_nonzero_factor_cancels_only_in_equality():
    original = model([Variable('a', -1, 1), Variable('x', -1, 1)], [
        (poly((1, 'a', 'a'), (-1,)), '=='),
        (poly((1, 'a', 'x')), '=='),
    ], poly((1, 'x')))
    reduced = reduce_model(original, fractional=False)
    check_bounds(original, reduced, (0, 0))
    inequality = model(original.variables, [
        (original.constraints[0].expression, '=='),
        (poly((1, 'a', 'x')), '<='),
    ], poly((1, 'x')))
    check_bounds(inequality, reduce_model(inequality), (-1, 1))


def test_positive_affine_factor_can_be_cancelled():
    original = model([Variable('a'), Variable('x', -1, 1)], [
        (poly((1, 'a', 'x'), (1, 'x')), '<='),
    ], poly((1, 'x')))
    reduced = reduce_model(original, fractional=False)
    assert reduced.model.is_linear
    check_bounds(original, reduced, (-1, 0))


def fractional_model(positive=True):
    return model([Variable('x'), Variable('y'), Variable('theta', -2, 2)], [
        (poly((1, 'x'), (1, 'y'), (-1,)), '<='),
        (poly((1, 'x'), (-1, 'theta', 'y'), (-1 if positive else 0, 'theta')), '=='),
    ], poly((1, 'theta')))


def test_charnes_cooper_preserves_bounds_and_restores_original_witnesses():
    original = fractional_model()
    reduced = reduce_model(original)
    assert reduced.model.is_linear
    assert any(s[0] == 'charnes_cooper' for s in reduced.steps)
    check_bounds(original, reduced, (0, 1))


def test_fractional_zero_denominator_is_not_silently_excluded():
    original = fractional_model(False)
    reduced = reduce_model(original)
    assert not any(s[0] == 'charnes_cooper' for s in reduced.steps)
    check_bounds(original, reduced, (-2, 2))


def test_fractional_denominator_sign_and_target_domain_are_preserved():
    original = model([Variable('x'), Variable('theta', -F(3, 4), -F(1, 4))], [
        (poly((1, 'x'), (1, 'theta'), (1, 'theta', 'x')), '=='),
    ], poly((1, 'theta')))
    reduced = reduce_model(original)
    assert reduced.model.is_linear
    check_bounds(original, reduced, (-.5, -.25))


def test_unbounded_denominator_is_not_replaced_by_zero_scale_closure():
    original = model([Variable('x', 1, None), Variable('theta', None, None)], [
        (poly((1, 'theta', 'x'), (-1,)), '=='),
    ], poly((1, 'theta')))
    reduced = reduce_model(original)
    assert not any(s[0] == 'charnes_cooper' for s in reduced.steps)


def test_contradictory_equalities_are_not_rounded_away():
    original = model([Variable('x')], [
        (poly((1, 'x'), (-F(1, 2),)), '=='),
        (poly((1, 'x'), (-F(1, 2)-F(1, 10**20),)), '=='),
    ], poly((1, 'x')))
    reduced = reduce_model(original)
    assert reduced.model == original
    assert ('skipped_inconsistent_exact_equalities',) in reduced.steps


def test_out_of_domain_fixed_values_remain_infeasible():
    original = model([Variable('x'), Variable('y')], [
        (poly((1, 'x'), (-2,)), '=='),
        (poly((1, 'x', 'y')), '=='),
    ], poly((1, 'y')))
    reduced = reduce_model(original)
    result = reduced.solve(verbose=False)
    assert result.lower.end == result.upper.end == 0


def test_reduction_budget_skips_expansion_without_relaxation():
    original = model([Variable('a'), Variable('b'), Variable('x')], [
        (poly((1, 'a'), (1, 'b'), (-1,)), '=='),
        (poly((1, 'a', 'a', 'x'), (-F(1, 10),)), '<='),
    ], poly((1, 'x')))
    reduced = reduce_model(original, max_terms=1, fractional=False)
    assert reduced.model == original
    for a, b, x in product([0, F(1, 2), 1], repeat=3):
        values = {'a': a, 'b': b, 'x': x}
        assert [c.expression.evaluate(values) for c in reduced.model.constraints] == [
            c.expression.evaluate(values) for c in original.constraints]


def test_program_opt_in_retains_declarations_and_reconstructs_dgps():
    from autobounds import Program
    program = Program()
    program.parameters = ['x', 'objvar']
    program.constraints = [[['x'], ['-1', 'objvar', 'x'], ['-1', 'objvar'], ['==']]]
    before = program.compile()
    _, dgps = program.run_scip(verbose=False, linearize=True, return_dgps=True)
    assert program.compile() == before
    assert program.last_reduction.model.is_linear
    for value in dgps.values():
        assert set(value['values']) == {'x', 'objvar'}
        check_witness(before, value['values'])

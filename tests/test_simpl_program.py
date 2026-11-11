"""Mathematical equivalence checks for conservative polynomial simplification."""
from copy import deepcopy
from fractions import Fraction
import random

import pytest
from autobounds import Program
from autobounds.polynomial import collect_terms
from autobounds.ProgramUtils import change_constraint_parameter_value


def make_program(constraints, parameters=('x', 'y', 'objvar')):
    program = Program()
    program.parameters = list(parameters)
    program.constraints = deepcopy(constraints)
    return program


def assert_same_bounds(before, after, expected):
    for program in (before, after):
        result = program.run_scip(verbose=False)
        for endpoint, value in zip(result[:2], expected):
            assert endpoint['primal'] == pytest.approx(value, abs=1e-6)
            assert endpoint['dual'] == pytest.approx(value, abs=1e-6)


def test_simplification_combines_terms_and_preserves_fixed_variable_equations():
    rows = [[['x'], ['-0.5'], ['==']],
            [['x', 'y'], ['2', 'y', 'x'], ['-1', 'x', 'y'], ['-1', 'objvar'], ['==']],
            [['x'], ['-0.5'], ['==']]]
    before, after = make_program(rows), make_program(rows)
    report = after.simplify()
    assert report['constraints_before'] == 3
    assert report['constraints_after'] == 2
    assert report['terms_after'] < report['terms_before']
    assert report['fixed_parameters'] == ('x',)
    assert after.parameters == before.parameters
    assert_same_bounds(before, after, (0, 1))
    rows_after = deepcopy(after.constraints)
    after.simplify()
    assert after.constraints == rows_after


@pytest.mark.parametrize('rows', [
    [[['x'], ['-2'], ['==']]],  # x is still constrained to [0, 1].
    [[['x'], ['-0.2'], ['==']], [['x'], ['-0.3'], ['==']]],
    [[['x'], ['==']], [['x'], ['-0.1'], ['>=']]],
])
def test_simplification_preserves_infeasibility(rows):
    rows = rows + [[['objvar'], ['==']]]
    for simplify in (False, True):
        program = make_program(rows)
        if simplify:
            program.simplify()
        result = program.run_scip(verbose=False)
        assert all(endpoint['end'] == 0 for endpoint in result[:2])


def test_simplification_does_not_treat_inequalities_as_equalities():
    rows = [[['x'], ['-0.25'], ['>=']],
            [['x'], ['-1', 'objvar'], ['==']]]
    before, after = make_program(rows), make_program(rows)
    assert after.simplify()['fixed_parameters'] == ()
    assert_same_bounds(before, after, (0.25, 1))


def test_fixed_variable_propagation_handles_chains_and_powers():
    rows = [[['x'], ['-2', 'y'], ['==']], [['y'], ['-0.25'], ['==']],
            [['x', 'x'], ['-1', 'objvar'], ['==']]]
    before, after = make_program(rows), make_program(rows)
    assert after.simplify()['fixed_parameters'] == ('objvar', 'x', 'y')
    assert_same_bounds(before, after, (0.25, 0.25))


def test_nonterminating_fixed_values_are_not_rounded():
    program = make_program([[['3', 'x'], ['-1'], ['==']]])
    report = program.simplify()
    assert report['fixed_parameters'] == ()
    assert ['3', 'x'] in program.constraints[0]


def test_collecting_terms_preserves_polynomial_exactly():
    def evaluate(row, values):
        total = Fraction(0)
        for term in row[:-1]:
            product = Fraction(1)
            for token in term:
                product *= values[token] if token in values else Fraction(token)
            total += product
        return total

    rng = random.Random(739)
    for _ in range(50):
        row = [[str(rng.randint(-5, 5)), *rng.choices(['x', 'y'], k=rng.randrange(4))]
               for _ in range(12)] + [['<=']]
        program = make_program([row])
        program.simplify()
        for x, y in [(0, 1), (Fraction(3, 10), Fraction(7, 10)), (1, 1)]:
            values = {'x': x, 'y': y}
            if program.constraints:
                assert evaluate(row, values) == evaluate(program.constraints[0], values)
            else:
                assert evaluate(row, values) <= 0


def test_explicit_parameter_fix_substitutes_every_power_and_keeps_domain():
    row = [['x', 'x', 'y'], ['-1', 'objvar'], ['==']]
    result = change_constraint_parameter_value(row, 'x', 0.5)
    assert collect_terms(result, {'y', 'objvar'}) == {('y',): Fraction(1, 4), ('objvar',): -1}
    assert row[0] == ['x', 'x', 'y']
    program = make_program([row])
    program.optimize_add_param_value('x', 2)
    result = program.run_scip(verbose=False)
    assert all(endpoint['end'] == 0 for endpoint in result[:2])


def test_old_unsafe_simplifier_is_removed():
    import autobounds.ProgramUtils as utils
    assert not hasattr(Program, 'simplify_linear')
    for name in ('replace_linear', 'replace_expr_lin', 'replace_expr_lin_nonlin',
                 'parse_bounds', 'solve1', 'is_linear',
                 'test_string_numeric', 'test_string_numeric_list'):
        assert not hasattr(utils, name)


def test_simplification_of_observational_monotone_causal_model():
    import pandas as pd
    from autobounds import DAG, Bounder
    problem = Bounder(DAG('X -> Y'))
    problem.load_data(pd.DataFrame({
        'X': [0, 0, 1, 1], 'Y': [0, 1, 0, 1], 'prob': [.4, .1, .2, .3],
    }))
    problem.set_p_to_zero(problem.p('Y(X=0)=1&Y(X=1)=0'))
    problem.set_ate('X', 'Y')
    before, after = problem.write_program(), problem.write_program()
    report = after.simplify()
    assert report['terms_after'] < report['terms_before']
    assert report['constraints_after'] < report['constraints_before']
    assert_same_bounds(before, after, (.4, .4))

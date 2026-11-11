"""Zero support inference must preserve feasible sets and solver certificates."""
import io
from dataclasses import replace
from fractions import Fraction
from itertools import product
from operator import eq, ge, le
import random

import pytest

from autobounds import Bounder, DAG, Program, Q, MathematicalModel, Variable, Polynomial, Term, Constraint
from autobounds.backends.scip import ScipBackend
from autobounds.support import eliminate_zero_variables


def is_feasible(candidate, values):
    """Evaluate directly, independently of the substitution implementation."""
    for variable in candidate.variables:
        value = values[variable.name]
        if variable.lower is not None and value < variable.lower:
            return False
        if variable.upper is not None and value > variable.upper:
            return False
    comparisons = {'==': eq, '<=': le, '>=': ge}
    for row in candidate.constraints:
        total = 0
        for term in row.expression.terms:
            value = term.coefficient
            for name in term.variables:
                value *= values[name]
            total += value
        if not comparisons[row.relation](total, 0):
            return False
    return True


def model(rows, variables=None):
    variables = variables or (Variable('x'), Variable('y'))
    names = [v.name for v in variables]
    return MathematicalModel(variables, tuple(
        Constraint(Polynomial.from_tokens(terms, names), relation)
        for terms, relation in rows), Polynomial((Term(1, ('x',)),)))


@pytest.mark.parametrize('terms,relation,zeros', [
    ([['2', 'x'], ['3', 'y']], '==', {'x', 'y'}),
    ([['x'], ['y']], '<=', {'x', 'y'}),
    ([['-1', 'x'], ['-2', 'y']], '>=', {'x', 'y'}),
    ([['x'], ['y']], '>=', set()),
    ([['x'], ['-1', 'y']], '==', set()),
    ([['x', 'y']], '==', set()),
    ([['x'], ['y'], ['-0.000000000000000000000000000001']], '<=', set()),
])
def test_only_proven_zeros_are_removed(terms, relation, zeros):
    original = model([(terms, relation)])
    reduced = eliminate_zero_variables(original)
    assert set(reduced.eliminated_values) == zeros
    assert eliminate_zero_variables(reduced) == reduced
    assert len(original.variables) == 2


def test_zero_propagation_and_nonzero_factor_proofs():
    original = model([([['x', 'y']], '=='), ([['y'], ['-0.5']], '==')])
    assert eliminate_zero_variables(original).eliminated_values == {'x': 0}
    bounded = model([([['x', 'y']], '==')], (Variable('x'), Variable('y', .1, 1)))
    assert eliminate_zero_variables(bounded).eliminated_values == {'x': 0}
    chain = model([([['x']], '=='), ([['x'], ['-1', 'y']], '==')])
    assert eliminate_zero_variables(chain).eliminated_values == {'x': 0, 'y': 0}


def test_signed_domains_and_squares():
    variables = (Variable('x', -1, 1), Variable('y', -1, 1))
    contrast = model([([['x'], ['y']], '==')], variables)
    assert eliminate_zero_variables(contrast) == contrast
    squares = model([([['x', 'x'], ['y', 'y']], '<=')], variables)
    assert set(eliminate_zero_variables(squares).eliminated_values) == {'x', 'y'}


def test_parameters_and_out_of_domain_zeros_are_not_removed():
    original = model([([['x']], '==')], (Variable('x', .2, 1),))
    assert eliminate_zero_variables(original) == original
    symbolic = model([([['x']], '==')], (Variable('x', kind='parameter'),))
    assert eliminate_zero_variables(symbolic) == symbolic


def test_elimination_preserves_contradictions():
    original = model([([['x'], ['y']], '=='),
                      ([['x'], ['y'], ['-1']], '==')])
    reduced = eliminate_zero_variables(original)
    assert not reduced.variables
    assert reduced.constraints == (Constraint(Polynomial((Term(-1),))),)
    for candidate in (original, reduced):
        native = ScipBackend.build_model(candidate, 'minimize')
        native.hideOutput()
        native.optimize()
        assert str(native.getStatus()) == 'infeasible'


def test_compilation_keeps_source_support_for_relaxation():
    program = Program()
    program.parameters = ['x', 'y', 'objvar']
    program.constraints = [[['x'], ['y'], ['<=']],
                           [['objvar'], ['-1', 'x'], ['==']]]
    assert set(program.compile().eliminated_values) == {'x', 'y', 'objvar'}
    assert len(program.parameters) == 3
    program.constraints[0].insert(2, ['-0.1'])
    assert program.compile().eliminated_values == {}


def test_simplify_and_recompile_restore_zero_objective():
    program = Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], ['==']]]
    report = program.simplify()
    assert report['eliminated_parameters'] == ('objvar',)
    bounds, dgps = program.run_scip(verbose=False, return_dgps=True)
    for endpoint in bounds[:2]:
        assert endpoint['primal'] == pytest.approx(0)
        assert endpoint['dual'] == pytest.approx(0)
    assert dgps['lower']['values']['objvar'] == 0
    native = ScipBackend.build_model(program.compile(limits=(.1, 1)), 'minimize')
    native.hideOutput()
    native.optimize()
    assert str(native.getStatus()) == 'infeasible'


def test_monotonicity_reduces_support_and_preserves_both_bounds():
    problem = Bounder(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'))
    problem.add_assumption(problem.p('X(Z=0)=1&X(Z=1)=0'), '<=', 0)
    problem.load_data(io.StringIO(
        'Z,X,Y,prob\n0,0,0,0.125\n0,0,1,0.125\n0,1,0,0.125\n'
        '0,1,1,0.125\n1,0,0,0.125\n1,0,1,0.125\n1,1,0,0.125\n1,1,1,0.125'))
    problem.set_ate('X', 'Y')
    program = problem.write_program()
    original = program.compile(eliminate_zeros=False)
    reduced = program.compile()
    assert len(original.variables) - len(reduced.variables) == 4
    assert set(reduced.eliminated_values) == {f'X10.Y{y}' for y in ('00', '01', '10', '11')}
    for sense in ('minimize', 'maximize'):
        outcomes = []
        for candidate in (original, reduced):
            native = ScipBackend.build_model(candidate, sense)
            native.hideOutput()
            native.optimize()
            outcomes.append((native.getObjVal(), native.getDualbound()))
        assert outcomes[0] == pytest.approx(outcomes[1])
        expected = -.5 if sense == 'minimize' else .5
        assert outcomes[1] == pytest.approx((expected, expected))
    bounds, dgps = program.run_scip(verbose=False, return_dgps=True)
    for endpoint in dgps.values():
        assert endpoint['status'] == 'ok'
        assert all(endpoint['values'][name] == 0 for name in reduced.eliminated_values)


def test_q_zero_assumptions_preserve_signed_contrasts_and_products():
    problem = Bounder(DAG('X -> Y'))
    problem.set_p_to_zero(Q([(1, ['Y00']), (-1, ['Y11'])]))
    problem.set_p_to_zero(Q([(1, ['X0', 'Y01'])]))
    assert not problem.zero_parameters
    assert not problem.compile().eliminated_values
    with pytest.raises(ValueError, match='Unknown strata'):
        problem.set_p_to_zero(['typo'])


@pytest.mark.parametrize('assumption_first', [True, False])
def test_observed_positive_root_conflicts_with_zero_assumption(assumption_first):
    problem = Bounder(DAG('X -> Y'))
    if assumption_first:
        problem.set_p_to_zero(['X0'])
    problem.load_data(io.StringIO('X,Y,prob\n0,0,0.25\n0,1,0.25\n1,0,0.25\n1,1,0.25'))
    if not assumption_first:
        problem.set_p_to_zero(['X0'])
    problem.set_ate('X', 'Y')
    compiled = problem.compile()
    assert 'X0' not in compiled.eliminated_values
    native = ScipBackend.build_model(compiled, 'minimize')
    native.hideOutput()
    native.optimize()
    assert str(native.getStatus()) == 'infeasible'


@pytest.mark.parametrize('seed', [17, 43, 91])
def test_random_polynomial_reductions_preserve_exact_grid_feasibility(seed):
    rng = random.Random(seed)
    grid = tuple(Fraction(i, 2) for i in range(-2, 3))
    monomials = ((), ('x',), ('y',), ('x', 'x'), ('y', 'y'),
                 ('x', 'y'), ('x', 'x', 'y'), ('x', 'y', 'y'))
    domains = ((0, 1), (-1, 1), (-1, 0), (0, 0), (Fraction(1, 2), 1))
    reductions = 0
    for case in range(150):
        variables = tuple(Variable(name, *rng.choice(domains)) for name in ('x', 'y'))
        rows = tuple(Constraint(Polynomial(tuple(
            Term(rng.choice((-2, -1, 1, 2)), rng.choice(monomials))
            for _ in range(rng.randint(1, 3)))), rng.choice(('==', '<=', '>=')))
            for _ in range(rng.randint(1, 3)))
        original = MathematicalModel(variables, rows, Polynomial((
            Term(2, ('x', 'y')), Term(-1, ('x',)), Term(3, ('y',)))))
        reduced = eliminate_zero_variables(original)
        reductions += bool(reduced.eliminated_values)
        assert eliminate_zero_variables(reduced) == reduced
        for point in product(grid, repeat=2):
            values = dict(zip(('x', 'y'), point))
            projected = {v.name: values[v.name] for v in reduced.variables}
            has_zero_support = all(values[name] == 0 for name in reduced.eliminated_values)
            assert is_feasible(original, values) == (
                has_zero_support and is_feasible(reduced, projected)
            ), (seed, case, original, reduced, values)
            if has_zero_support:
                assert original.objective.evaluate(values) == reduced.objective.evaluate(projected)
    assert reductions >= 20


def test_population_monotonicity_keeps_positive_defiers():
    problem = Bounder(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'))
    problem.add_assumption(problem.p('X(Z=1)=1') - problem.p('X(Z=0)=1'), '>=', 0)
    problem.set_ate('X', 'Y')
    compiled = problem.compile()
    assert compiled.eliminated_values == {}
    # Equal masses of compliers and defiers satisfy mean monotonicity.
    witness = {v.name: Fraction(1, 16) for v in compiled.variables}
    witness.update(Z0=Fraction(1, 2), Z1=Fraction(1, 2), objvar=0)
    assert is_feasible(compiled, witness)
    assert all(witness[f'X10.Y{y}'] > 0 for y in ('00', '01', '10', '11'))


def test_binding_sensitivity_parameter_distinguishes_zero_from_positive():
    original = model([([['x'], ['y'], ['-1', 'theta']], '<=')],
                     (Variable('x'), Variable('y'), Variable('theta', kind='parameter')))
    assert eliminate_zero_variables(original).eliminated_values == {}
    exact = eliminate_zero_variables(original.bind({'theta': 0}))
    relaxed = eliminate_zero_variables(original.bind({'theta': Fraction(1, 10**30)}))
    assert exact.eliminated_values == {'x': 0, 'y': 0}
    assert relaxed.eliminated_values == {}
    assert is_feasible(relaxed, {'x': Fraction(1, 10**30), 'y': 0})


def test_zero_factor_linearizes_model_and_preserves_solver_certificates():
    original = model([([['x']], '=='),
                      ([['y'], ['-0.25']], '>='),
                      ([['x', 'y', 'y'], ['y'], ['-0.75']], '<=')])
    original = replace(original, objective=Polynomial((
        Term(1, ('y',)), Term(2, ('x', 'y', 'y')))))
    reduced = eliminate_zero_variables(original)
    assert not original.is_linear
    assert reduced.is_linear
    for candidate in (original, reduced):
        for sense, expected in (('minimize', .25), ('maximize', .75)):
            native = ScipBackend.build_model(candidate, sense)
            native.hideOutput()
            native.optimize()
            assert native.getObjVal() == pytest.approx(expected)
            assert native.getDualbound() == pytest.approx(expected)
    backend = ScipBackend()
    result = backend.solve(original, verbose=False, return_dgps=True)
    assert backend.compiled_model.eliminated_values == {'x': 0}
    for endpoint, expected in ((result.lower, .25), (result.upper, .75)):
        assert endpoint.primal == pytest.approx(expected)
        assert endpoint.dual == pytest.approx(expected)
    for witness in result.dgps.values():
        assert witness['status'] == 'ok'
        assert witness['values']['x'] == 0
        assert is_feasible(original, witness['values'])


@pytest.mark.parametrize('leave_free_variable', [False, True])
def test_infeasible_reduced_model_does_not_fabricate_a_zero_witness(leave_free_variable):
    original = model([([['x'], ['y']], '=='),
                      ([['x'], ['y'], ['-1']], '==')])
    if leave_free_variable:
        original = replace(original, variables=original.variables + (Variable('z'),))
    result = ScipBackend().solve(original, verbose=False, return_dgps=True)
    for witness in result.dgps.values():
        assert witness['status'] == 'no_solution'
        assert witness['values'] == {}


def test_zero_domain_and_binding_preserve_reconstruction_metadata():
    original = model([([['y'], ['-1', 'theta']], '==')],
                     (Variable('x', 0, 0), Variable('y'),
                      Variable('theta', kind='parameter')))
    reduced = eliminate_zero_variables(original)
    bound = reduced.bind({'theta': Fraction(1, 2)})
    assert bound.eliminated_values == {'x': 0}
    values = {'y': Fraction(1, 2)}
    restored = bound.restore_solution(values)
    assert restored == {'x': 0, 'y': Fraction(1, 2)}
    assert values == {'y': Fraction(1, 2)}
    assert is_feasible(original.bind({'theta': Fraction(1, 2)}), restored)

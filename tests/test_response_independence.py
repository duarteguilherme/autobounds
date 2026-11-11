"""Overlapping latent roots retain response independence inside a district."""
from itertools import product

import pandas as pd
import pytest

from autobounds import DAG, Bounder
from autobounds.backends.scip import ScipBackend


OVERLAPPING = 'Z -> D, D -> Y, U -> D, U -> Y, U2 -> Z, U2 -> D'
CONFOUNDED = 'Z -> D, D -> Y, U -> Z, U -> D, U -> Y'


def make_problem(edges=OVERLAPPING, unob='U,U2'):
    return Bounder(DAG(edges, unob=unob))


def solve_fixture(table, weights, *, confounded=False, conditional=False,
                  ordinary_iv=False):
    problem = make_problem(CONFOUNDED, 'U') if confounded else make_problem()
    if ordinary_iv:
        problem = make_problem('Z -> D, D -> Y, U -> D, U -> Y', 'U')
    data = pd.DataFrame([
        (z, d, y, table[z][2*d+y] * (1 if conditional else weights[z]))
        for z, d, y in product(range(2), repeat=3)
    ], columns=['Z', 'D', 'Y', 'prob'])
    problem.load_data(data, cond=['Z'] if conditional else [])
    if conditional:
        # Separately supplied instrument prevalence must remain a constraint,
        # not be inferred from a table of conditional probabilities.
        problem.load_data(pd.DataFrame({'Z': [0, 1], 'prob': weights}))
    problem.set_estimand(problem.p('Y(D=1)=1'))
    result = ScipBackend().solve(problem.compile(), verbose=False, theta=1e-8,
                                 maxtime=20)
    return result


def assert_bounds(result, expected):
    for endpoint, value in zip((result.lower, result.upper), expected):
        assert endpoint.primal == pytest.approx(value, abs=2e-6)
        assert endpoint.dual == pytest.approx(value, abs=2e-6)


def outcome_independence_oracle(table, weights):
    """Independent LP over actual D and joint (Y(0),Y(1)), without D strata."""
    from pyscipopt import Model, quicksum
    bounds = []
    for sense in ('minimize', 'maximize'):
        model = Model()
        model.hideOutput()
        states = list(product(range(2), repeat=4))  # z, d, y0, y1
        q = {s: model.addVar(lb=0, ub=1) for s in states}
        for z, d, y in product(range(2), repeat=3):
            model.addCons(quicksum(q[s] for s in states
                                  if s[0] == z and s[1] == d and s[2+d] == y)
                          == weights[z] * table[z][2*d+y])
        for y0, y1 in product(range(2), repeat=2):
            model.addCons(weights[1] * quicksum(q[s] for s in states
                                               if s[0] == 0 and s[2:] == (y0, y1))
                          == weights[0] * quicksum(q[s] for s in states
                                                  if s[0] == 1 and s[2:] == (y0, y1)))
        model.setObjective(quicksum(q[s] for s in states if s[3] == 1), sense)
        model.optimize()
        assert model.getStatus() == 'optimal'
        assert model.getObjVal() == pytest.approx(model.getDualbound())
        bounds.append(model.getObjVal())
    return bounds


@pytest.mark.parametrize('table,weights,expected', [
    ([[.1, .2, .2, .5], [.25, .25, .4, .1]], [.25, .75], [.5, .6]),
    # Joint outcome independence is stronger than the marginal assumptions
    # underlying Robins-Manski: here RM is [.6,.9], but sharp is [.7,.8].
    ([[.1, .7, .1, .1], [.3, 0, .1, .6]], [.5, .5], [.7, .8]),
])
def test_overlapping_latents_against_robins_manski_and_confounding(table, weights, expected):
    rm = (max(row[3] for row in table), 1-max(row[2] for row in table))
    confounded = (sum(w*row[3] for w, row in zip(weights, table)),
                  1-sum(w*row[2] for w, row in zip(weights, table)))
    assert expected == pytest.approx(outcome_independence_oracle(table, weights))
    assert_bounds(solve_fixture(table, weights), expected)
    assert_bounds(solve_fixture(table, weights, ordinary_iv=True), expected)
    assert_bounds(solve_fixture(table, weights, confounded=True), confounded)
    assert confounded[0] < expected[0] <= expected[1] < confounded[1]
    assert rm[0] <= expected[0] and expected[1] <= rm[1]


def test_conditional_data_with_separately_supplied_instrument_marginal():
    table = [[.1, .2, .2, .5], [.25, .25, .4, .1]]
    assert_bounds(solve_fixture(table, [.25, .75], conditional=True), [.5, .6])


def test_overlapping_components_use_transitive_closure():
    dag = DAG(OVERLAPPING, unob='U,U2')
    assert dag.find_c_components() == [{'Z', 'D', 'Y'}]
    assert dag.find_u_linked('Z') == {'Z', 'D'}  # direct neighbors stay direct


def test_independence_uses_three_sparse_equations_and_is_idempotent():
    problem = make_problem()
    problem.set_estimand(problem.p('Y(D=1)=1'))
    before = len(problem.constraints)
    first = problem.compile()
    problem.add_indep_constraints()
    second = problem.compile()
    assert first == second
    assert len(first.constraints) == len(set(first.constraints))
    assert len(problem.constraints) == before
    quadratic = [c for c in first.constraints if c.expression.degree == 2]
    assert len(quadratic) == (2-1)*(4-1)
    assert all(sum(len(t.variables) == 2 for t in c.expression.terms) == 1
               for c in quadratic)
    assert len([v for v in first.variables if v.name.startswith('_ab_indep_')]) == 4


def test_one_common_latent_has_no_extra_constraints():
    problem = make_problem(CONFOUNDED, 'U')
    assert problem._independence_templates == ()


def test_independent_blocks_include_joint_responses_not_only_pairs():
    problem = make_problem('U -> A, U -> B, V -> B, V -> C, W -> B, W -> E', 'U,V,W')
    templates = problem._independence_templates
    # A,C,E are mutually independent. Maximal blocks include A independent
    # of (C,E), ruling out pairwise-independent parity distributions.
    assert len(templates) == 3
    assert all(sorted((len(left), len(right))) == [1, 3]
               for _, left, right, _ in templates)


def test_joint_data_linearizes_independence_without_expanding_products():
    problem = make_problem()
    table = [[.1, .2, .2, .5], [.25, .25, .4, .1]]
    problem.load_data(pd.DataFrame([
        (z, d, y, .5 * table[z][2*d+y])
        for z, d, y in product(range(2), repeat=3)
    ], columns=['Z', 'D', 'Y', 'prob']))
    problem.set_estimand(problem.p('Y(D=1)=1'))
    model = problem.compile()
    assert all(c.expression.degree <= 1 for c in model.constraints)
    # Three unknown Y-response marginals; the Z marginal is a known constant.
    assert len([v for v in model.variables if v.name.startswith('_ab_indep_')]) == 3


def test_conditional_data_alone_does_not_linearize_unknown_prevalence():
    problem = make_problem()
    table = [[.1, .2, .2, .5], [.25, .25, .4, .1]]
    problem.load_data(pd.DataFrame([
        (z, d, y, table[z][2*d+y])
        for z, d, y in product(range(2), repeat=3)
    ], columns=['Z', 'D', 'Y', 'prob']), cond=['Z'])
    problem.set_estimand(problem.p('Y(D=1)=1'))
    assert sum(c.expression.degree == 2 for c in problem.compile().constraints) == 3


def test_zero_instrument_cell_needs_no_division():
    table = [[.1, .2, .2, .5], [.25, .25, .4, .1]]
    assert_bounds(solve_fixture(table, [0, 1]), [.1, .6])


def test_excluded_outcome_response_still_satisfies_independence():
    problem = make_problem()
    problem.add_constraint(problem.p('Y(D=0)=0&Y(D=1)=1'))
    problem.set_estimand(problem.p('Y(D=1)=1'))
    model = problem.compile()
    assert not any('Y01' in v.name for v in model.variables)
    assert model == problem.compile()


def test_independence_constraints_reject_pairwise_independent_parity():
    problem = make_problem('U -> A, U -> B, V -> B, V -> C, W -> B, W -> E', 'U,V,W')
    program = problem.write_program()
    values = {name: 0 for name in program.parameters}
    for name in problem.canModel.parameters:
        tokens = dict((t[0], int(t[1:])) for t in name.split('.'))
        if tokens['B'] == 0 and tokens['E'] == (tokens['A'] ^ tokens['C']):
            values[name] = .25
    # Assign each introduced marginal its actual sum from its defining row.
    for row in program.constraints:
        if len(row) >= 2 and row[-2][0] == '-1' and len(row[-2]) == 2:
            target = row[-2][1]
            if target.startswith('_ab_indep_'):
                values[target] = sum(values[t[0]] for t in row[:-2])
    violations = [c.expression.evaluate(values) for c in program.compile().constraints]
    assert any(abs(v) > .01 for v in violations)


def test_nonuniform_independent_latent_generators_remain_feasible():
    problem = make_problem('U -> A, U -> B, V -> B, V -> C, W -> B, W -> E', 'U,V,W')
    program = problem.write_program()
    from fractions import Fraction
    values = {name: Fraction(0) for name in program.parameters}
    for name in problem.canModel.parameters:
        tokens = dict((t[0], int(t[1:])) for t in name.split('.'))
        if tokens['B'] == (tokens['A'] ^ tokens['C'] ^ tokens['E']):
            value = Fraction(1)
            for node, probability in [('A', Fraction(1, 3)),
                                      ('C', Fraction(2, 5)),
                                      ('E', Fraction(3, 7))]:
                value *= probability if tokens[node] else 1-probability
            values[name] = value
    for row in program.constraints:
        if len(row) >= 2 and row[-2][0] == '-1' and len(row[-2]) == 2:
            target = row[-2][1]
            if target.startswith('_ab_indep_'):
                values[target] = sum(values[t[0]] for t in row[:-2])
    assert all(c.expression.evaluate(values) == 0 for c in program.compile().constraints)


def test_nonbinary_instrument_uses_reduced_constraint_basis():
    problem = Bounder(DAG(OVERLAPPING, unob='U,U2'), number_values={'Z': 3})
    problem.set_estimand(problem.p('Y(D=1)=1'))
    assert sum(c.expression.degree == 2 for c in problem.compile().constraints) == 6


def test_all_strata_excluded_preserves_infeasibility():
    problem = make_problem()
    problem.set_p_to_zero(problem.canModel.parameters)
    model = problem.compile()
    assert any(c.expression.degree == 0 and c.expression.constant != 0
               and c.relation == '==' for c in model.constraints)


def test_replay_reuses_templates_but_recomputes_known_marginals():
    problem = make_problem()
    problem.set_estimand(problem.p('Y(D=1)=1'))
    replay = problem._copy_for_replay()
    assert replay._independence_templates is problem._independence_templates
    replay.load_data(pd.DataFrame({'Z': [0, 1], 'prob': [.25, .75]}))
    assert all(c.expression.degree <= 1 for c in replay.compile().constraints)
    assert any(c.expression.degree == 2 for c in problem.compile().constraints)

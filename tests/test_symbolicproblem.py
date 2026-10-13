"""Exact symbolic certificates checked against independently solved primal LPs."""

import builtins
from itertools import product

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import linprog

from autobounds.autobounds import DAG, LinearResponseModel, SymbolicProblem


def assert_certificates(model, result):
    matrix = sp.Matrix([(1,) * len(model.target)] + list(model.matrix))
    target = sp.Matrix(model.target)
    for certificate in result.lower_certificates:
        assert all(value <= 0 for value in matrix.T * sp.Matrix(certificate) - target)
    for certificate in result.upper_certificates:
        assert all(value >= 0 for value in matrix.T * sp.Matrix(certificate) - target)
    assert result.sharp and result.complete


def assert_primal_dual_agree(model, result, q):
    incidence = np.asarray(model.matrix, dtype=float).reshape(-1, len(q))
    probabilities = incidence @ q
    matrix = np.vstack([np.ones(len(q)), incidence])
    rhs = np.r_[1, probabilities]
    target = np.asarray(model.target, dtype=float)
    lower = linprog(target, A_eq=matrix, b_eq=rhs, bounds=(0, None), method='highs')
    upper = linprog(-target, A_eq=matrix, b_eq=rhs, bounds=(0, None), method='highs')
    assert lower.success and upper.success
    actual = result.evaluate(dict(zip(model.labels, probabilities)))
    assert actual == pytest.approx((lower.fun, -upper.fun), abs=1e-9)


@pytest.fixture
def iv_problem():
    return SymbolicProblem(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'))


@pytest.fixture
def cdd_backend():
    pytest.importorskip('cdd.gmp')


def test_nested_events_recover_subtraction_certificate(cdd_backend):
    model = LinearResponseModel([[1, 1, 0], [0, 1, 0]], [1, 0, 0], ['a', 'b'])
    result = model.solve()
    a, b = model.symbols
    assert result.lower == result.upper == a - b
    assert result.evaluate({'a': .7, 'b': .2}) == pytest.approx((.5, .5))
    assert_certificates(model, result)
    assert_primal_dual_agree(model, result, np.array([.5, .2, .3]))


def test_fractional_certificates_recover_bound_missed_by_binary_covers(cdd_backend):
    model = LinearResponseModel([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]],
                                [1, 1, 1, 0])
    result = model.solve()
    assert result.lower == result.upper == sum(model.symbols) / 2
    assert result.evaluate(dict.fromkeys(model.labels, .2)) == pytest.approx((.3, .3))
    assert_certificates(model, result)
    assert_primal_dual_agree(model, result, np.array([.1, .1, .1, .7]))


def test_signed_rational_target_without_data_has_sharp_extremes(cdd_backend):
    model = LinearResponseModel([], [sp.Rational(-3, 2), 0, sp.Rational(2, 3)])
    result = model.solve()
    assert (result.lower, result.upper) == (sp.Rational(-3, 2), sp.Rational(2, 3))
    assert_certificates(model, result)


def test_dependent_rows_and_repeated_response_types(cdd_backend):
    model = LinearResponseModel([[1, 0, 1], [0, 1, 0], [1, 0, 1]], [0, 2, 1])
    result = model.solve()
    assert len(result.compatibility_equalities) == 2
    assert_certificates(model, result)
    assert_primal_dual_agree(model, result, np.array([.2, .3, .5]))


def test_random_small_models_match_primal_including_boundary_data(cdd_backend):
    rng = np.random.default_rng(503)
    for _ in range(8):
        model = LinearResponseModel(rng.integers(0, 2, (3, 6)),
                                    rng.integers(-3, 4, 6))
        result = model.solve()
        assert_certificates(model, result)
        assert_primal_dual_agree(model, result, rng.dirichlet(np.ones(6)))
        assert_primal_dual_agree(model, result, np.eye(6)[0])


@pytest.mark.parametrize('ate', [False, True])
def test_iv_graph_builds_sharp_marginal_and_ate_bounds(iv_problem, ate, cdd_backend):
    problem = iv_problem
    problem.load_data('X,Y', cond='Z')
    if ate:
        problem.set_ate('X', 'Y')
    else:
        problem.set_estimand(problem.p('Y(X=1)=1'))
    model = problem.write_program()
    assert len(model.target) == 16
    assert len(model.matrix) == 8
    assert all(sum(model.matrix[i][s] for i in range(4)) == 1 for s in range(16))
    result = model.solve()
    assert_certificates(model, result)
    rng = np.random.default_rng(22)
    for _ in range(3):
        assert_primal_dual_agree(model, result, rng.dirichlet(np.ones(16)))
    assert_primal_dual_agree(model, result, np.eye(16)[0])
    if not ate:
        # Paper section 5.3: only the larger ordinary cover gets upper zero.
        values = [.5, 0, .5, 0, 0, .5, .5, 0]
        assert result.evaluate(dict(zip(model.labels, values))) == pytest.approx((0, 0))


def test_monotonicity_excludes_defier_types(iv_problem, cdd_backend):
    iv_problem.load_data('X,Y', cond='Z')
    iv_problem.set_ate('X', 'Y')
    iv_problem.set_p_to_zero(iv_problem.p('X(Z=0)=1&X(Z=1)=0'))
    model = iv_problem.write_program()
    assert len(model.target) == 12
    result = model.solve()
    assert_certificates(model, result)
    assert_primal_dual_agree(model, result, np.ones(12) / 12)


def test_interventional_data_and_multivalued_instrument(cdd_backend):
    problem = SymbolicProblem(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'), {'Z': 3})
    problem.load_data('X,Y', cond='Z')
    problem.load_data('Y', do='X=1')
    problem.set_estimand('Y(X=1)=1')
    model = problem.write_program()
    assert len(model.target) == 32
    assert len(model.matrix) == 14
    assert model.matrix[-1] == model.target
    result = problem.solve()
    assert result.lower == result.upper
    assert_certificates(model, result)
    assert_primal_dual_agree(model, result, np.ones(32) / 32)


def test_rejects_nonlinear_queries_and_unsupported_latent_structure(iv_problem):
    with pytest.raises(ValueError, match='nonlinear'):
        iv_problem.set_estimand(iv_problem.p('Y(X=1)=1') * iv_problem.p('Y(X=0)=1'))
    with pytest.raises(ValueError, match='independent observed roots'):
        iv_problem.load_data('Y', cond='X')
    with pytest.raises(ValueError, match='nonlinear restrictions'):
        SymbolicProblem(DAG('U -> X, U -> Y, V -> Y, V -> Z', unob='U,V'))
    with pytest.raises(ValueError, match='nonlinear'):
        iv_problem.load_data('Z,X,Y')


def test_rejects_multiple_components_instead_of_relaxing_independence(iv_problem):
    iv_problem.set_estimand('Y(X=1)=1')
    iv_problem.load_data('Z')
    with pytest.raises(ValueError, match='multiple independent'):
        iv_problem.write_program()


def test_individual_data_labels_are_retained_and_duplicates_are_safe(iv_problem, cdd_backend):
    event = iv_problem.p('Y(X=1)=1')
    iv_problem.add_data(event, label='trial')
    iv_problem.add_data(event, label='trial')
    iv_problem.set_estimand(event)
    result = iv_problem.solve()
    assert result.lower == result.upper == sp.Symbol('trial', real=True)
    with pytest.raises(ValueError, match='different event'):
        iv_problem.add_data(iv_problem.p('Y(X=0)=1'), label='trial')
    with pytest.raises(ValueError, match='Missing probabilities'):
        result.evaluate({})


def test_invalid_declarations_fail_early(iv_problem):
    with pytest.raises(ValueError, match='out-of-range'):
        iv_problem.p('Y(X=2)=1')
    with pytest.raises(ValueError, match='Set an estimand'):
        iv_problem.solve()
    with pytest.raises(ValueError, match='event indicators'):
        LinearResponseModel([[2, 0]], [0, 1])
    with pytest.raises(ValueError, match='distinct label'):
        LinearResponseModel([[0, 1], [1, 0]], [0, 1], ['p', 'p'])
    iv_problem.set_estimand('Y(X=1)=1')
    iv_problem.set_p_to_zero(1)
    with pytest.raises(ValueError, match='At least one admissible'):
        iv_problem.write_program()


def test_problem_does_not_mutate_inputs(iv_problem):
    graph = iv_problem.dag
    cardinalities = {'Z': 3}
    problem = SymbolicProblem(graph, cardinalities)
    assert cardinalities == {'Z': 3}
    problem.dag.add_v('NEW')
    assert 'NEW' not in graph.V


@pytest.mark.parametrize('catalogue_margins', [False, True])
def test_factorial_hull_matches_primal_at_interior_and_boundary_data(catalogue_margins):
    pytest.importorskip('cdd.gmp')
    problem = SymbolicProblem(DAG('A -> Y, B -> Y, U -> A, U -> B, U -> Y', unob='U'))
    problem.load_data('A,B,Y')
    for a, b in product(range(2), repeat=2):
        problem.load_data('Y', do=f'A={a},B={b}')
    if catalogue_margins:
        for margin in ('A,B', 'A,Y', 'B,Y'):
            problem.load_data(margin)
        problem.add_data(1, label='one')
    problem.set_estimand(problem.p('Y(A=1)=1'))
    model = problem.write_program()
    assert len(model.target) == 64
    result = problem.solve()
    assert result.method == 'exact_cdd_hull'
    assert_certificates(model, result)
    rng = np.random.default_rng(2026)
    for _ in range(5):
        assert_primal_dual_agree(model, result, rng.dirichlet(np.ones(64)))
    for i in (0, 10, 31, 63):
        assert_primal_dual_agree(model, result, np.eye(64)[i])
    if catalogue_margins:
        basic = LinearResponseModel(model.matrix[:16], model.target, model.labels[:16])
        basic_bounds = basic.solve()
        assert result.lower_certificates == tuple(
            cert + (0,) * 13 for cert in basic_bounds.lower_certificates)
        assert result.upper_certificates == tuple(
            cert + (0,) * 13 for cert in basic_bounds.upper_certificates)


@pytest.mark.parametrize('target', [(sp.Rational(3, 7),), (2, 2, 2)])
def test_hull_handles_zero_dimensional_models(target):
    pytest.importorskip('cdd.gmp')
    model = LinearResponseModel([(1,) * len(target), (0,) * len(target)], target)
    result = model.solve()
    assert result.lower == result.upper == target[0]
    assert len(result.compatibility_equalities) == 2
    assert_certificates(model, result)


def test_hull_retains_exact_large_rational_coefficients():
    pytest.importorskip('cdd.gmp')
    epsilon = sp.Rational(1, 10**40 + 7)
    model = LinearResponseModel([[1, 0, 0]], [0, 1 + epsilon, 1])
    result = model.solve()
    p, = model.symbols
    assert sp.expand(result.upper - (1 + epsilon) * (1 - p)) == 0
    assert result.lower == 1 - p
    assert not result.upper.atoms(sp.Float)
    assert_certificates(model, result)


def test_identified_target_preserves_observable_compatibility_inequalities():
    pytest.importorskip('cdd.gmp')
    model = LinearResponseModel([[0, 1]], [0, 2])
    result = model.solve()
    p, = model.symbols
    assert result.lower == result.upper == 2 * p
    assert all(bool(condition.subs(p, sp.Rational(1, 2)))
               for condition in result.compatibility_inequalities)
    for incompatible in (-1, 2):
        assert any(not bool(condition.subs(p, incompatible))
                   for condition in result.compatibility_inequalities)


def test_missing_cdd_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def without_cdd(name, *args, **kwargs):
        if name == 'cdd.gmp':
            raise ModuleNotFoundError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_cdd)
    model = LinearResponseModel([], [0, 1])
    with pytest.raises(ImportError, match=r'autobounds\[symbolic\]'):
        model.solve()


def test_backend_errors_do_not_return_partial_bounds(monkeypatch):
    cdd = pytest.importorskip('cdd.gmp')
    model = LinearResponseModel([], [0, 1])

    def fail_conversion(*args, **kwargs):
        raise RuntimeError('conversion failed')

    monkeypatch.setattr(cdd, 'polyhedron_from_matrix', fail_conversion)
    with pytest.raises(RuntimeError, match='conversion failed'):
        model.solve()

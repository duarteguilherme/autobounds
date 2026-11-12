"""Choose by model structure, and never hide a selected backend's failure."""
import pytest

from autobounds import DAG, LinearResponseModel, SymbolicProblem
from autobounds.parametric import ParametricResponseProgram


def simple(mode=None):
    p = SymbolicProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'), parametric=mode)
    p.load_data('X,Y')
    p.set_ate('X', 'Y')
    return p


def test_default_selects_fast_linear_solver():
    problem = simple()
    program = problem.write_program()
    assert isinstance(program, LinearResponseModel)
    assert program.compile() == simple(False).compile()
    assert problem.solver_selection['method'] == 'linear_response'


def test_explicit_override_still_selects_reduction():
    assert isinstance(simple(True).write_program(), ParametricResponseProgram)


def test_selection_reconsiders_new_assumptions():
    problem = simple()
    assert isinstance(problem.write_program(), LinearResponseModel)
    problem.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    assert isinstance(problem.write_program(), ParametricResponseProgram)
    assert problem.solver_selection['reason'] == 'homogeneity assumptions'


def test_failed_declaration_does_not_change_method():
    problem = simple()
    with pytest.raises(ValueError, match='Invalid event'):
        problem.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['nonsense'])
    assert isinstance(problem.write_program(), LinearResponseModel)


def test_selected_fast_backend_bug_propagates_without_fallback(monkeypatch):
    error = RuntimeError('injected fast-backend bug')
    def fail(*args, **kwargs):
        raise error
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected fallback to reduction')
    monkeypatch.setattr(LinearResponseModel, 'solve', fail)
    monkeypatch.setattr(ParametricResponseProgram, 'solve', forbidden)
    with pytest.raises(RuntimeError) as caught:
        simple().solve()
    assert caught.value is error


def test_selected_reduction_bug_propagates_without_fallback(monkeypatch):
    error = ValueError('injected reduction bug')
    def fail(*args, **kwargs):
        raise error
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected fallback to linear solver')
    monkeypatch.setattr(ParametricResponseProgram, 'solve', fail)
    monkeypatch.setattr(LinearResponseModel, 'solve', forbidden)
    problem = simple()
    problem.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    with pytest.raises(ValueError) as caught:
        problem.solve()
    assert caught.value is error


def test_unsupported_nonlinearity_remains_an_explicit_error():
    pytest.importorskip('cdd.gmp')
    problem = simple()
    event = problem.p('Y(X=1)=1')
    problem.set_estimand(event*event)
    assert isinstance(problem.write_program(), ParametricResponseProgram)
    with pytest.raises(ValueError, match='Unresolved nonlinear response products'):
        problem.solve()


def test_missing_independent_component_data_are_not_pooled():
    pytest.importorskip('cdd.gmp')
    problem = SymbolicProblem(DAG('Z -> X, Z -> Y, X -> Y, U -> X, U -> Y', unob='U'))
    problem.load_data('X,Y', cond='Z')
    problem.set_ate('X', 'Y')
    with pytest.raises(ValueError, match='independent components are not pooled'):
        problem.solve()


def test_forced_linear_mode_rejects_homogeneity():
    with pytest.raises(ValueError, match='forced parametric=False'):
        simple(False).add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    problem = simple()
    problem.add_homogeneity('Y(X=1)=1', 'Y(X=0)=1', strata=['X=0', 'X=1'])
    problem.parametric = False
    with pytest.raises(ValueError, match='forced parametric=False'):
        problem.write_program()


def test_invalid_mode_is_rejected():
    with pytest.raises(TypeError, match='parametric'):
        simple('auto')

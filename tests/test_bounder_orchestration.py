from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.causalProblem import causalProblem
from autobounds.autobounds.bounder import Bounder
import io
import numpy as np
import pytest


def test_default_bounder_registry():
    problem = causalProblem(DAG("D -> Y"))
    assert problem.list_bounders() == ["default"]
    assert isinstance(problem.get_bounder("default"), Bounder)
    assert problem.bounders == [problem.get_bounder("default")]


def test_add_and_get_named_bounder():
    manager = causalProblem(DAG("D -> Y"))
    extra = Bounder(DAG("Z -> X, X -> Y"))
    manager.add_bounder(extra, name="sensitivity")
    assert manager.list_bounders() == ["default", "sensitivity"]
    assert manager.get_bounder("sensitivity") is extra
    assert manager.bounders == [manager.get_bounder("default"), extra]


def test_new_bounder_factory():
    manager = causalProblem(DAG("D -> Y"))
    created = manager.new_bounder(name="ci_sample")
    assert isinstance(created, Bounder)
    assert manager.get_bounder("ci_sample") is created


def test_add_bounder_duplicate_name_requires_replace():
    manager = causalProblem(DAG("D -> Y"))
    manager.add_bounder(Bounder(DAG("Z -> X")), name="b1")
    try:
        manager.add_bounder(Bounder(DAG("A -> B")), name="b1")
        raised = False
    except ValueError:
        raised = True
    assert raised

    replacement = Bounder(DAG("A -> B"))
    manager.add_bounder(replacement, name="b1", replace=True)
    assert manager.get_bounder("b1") is replacement


def test_single_problem_wrappers_forward_to_default_bounder():
    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    # Wrapper should update default bounder state.
    assert problem.get_bounder("default").estimand is not None


def _build_iv_problem():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob="U")
    problem = causalProblem(dag)
    datafile = io.StringIO(
        """X,Y,Z,prob
0,0,0,0.05
0,0,1,0.05
0,1,0,0.1
0,1,1,0.1
1,0,0,0.15
1,0,1,0.15
1,1,0,0.2
1,1,1,0.2"""
    )
    problem.set_ate("X", "Y")
    problem.load_data(datafile)
    problem.add_prob_constraints()
    return problem


def test_causalproblem_solve_matches_default_bounder():
    cp = _build_iv_problem()
    bd = _build_iv_problem().get_bounder("default")

    res_cp = cp.solve(verbose_result=False, verbose_optimizer=False, maxtime=5)
    res_bd = bd.solve(verbose_result=False, verbose_optimizer=False, maxtime=5)

    assert res_cp["point lb dual"] == pytest.approx(res_bd["point lb dual"], abs=1e-6)
    assert res_cp["point ub dual"] == pytest.approx(res_bd["point ub dual"], abs=1e-6)
    assert res_cp["point lb primal"] == pytest.approx(res_bd["point lb primal"], abs=1e-6)
    assert res_cp["point ub primal"] == pytest.approx(res_bd["point ub primal"], abs=1e-6)


def test_default_bounder_solve_is_deterministic_with_seed():
    np.random.seed(2026)
    first = _build_iv_problem().solve(verbose_result=False, verbose_optimizer=False, maxtime=5)
    np.random.seed(2026)
    second = _build_iv_problem().solve(verbose_result=False, verbose_optimizer=False, maxtime=5)

    assert first["point lb dual"] == pytest.approx(second["point lb dual"], abs=1e-6)
    assert first["point ub dual"] == pytest.approx(second["point ub dual"], abs=1e-6)
    assert first["point lb primal"] == pytest.approx(second["point lb primal"], abs=1e-6)
    assert first["point ub primal"] == pytest.approx(second["point ub primal"], abs=1e-6)

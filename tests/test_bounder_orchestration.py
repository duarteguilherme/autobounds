from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.causalProblem import causalProblem
from autobounds.autobounds.bounder import Bounder


def test_default_bounder_registry():
    problem = causalProblem(DAG("D -> Y"))
    assert problem.list_bounders() == ["default"]
    assert problem.get_bounder("default") is problem
    assert problem.bounders == [problem]


def test_add_and_get_named_bounder():
    manager = causalProblem(DAG("D -> Y"))
    extra = Bounder(DAG("Z -> X, X -> Y"))
    manager.add_bounder(extra, name="sensitivity")
    assert manager.list_bounders() == ["default", "sensitivity"]
    assert manager.get_bounder("sensitivity") is extra
    assert manager.bounders == [manager, extra]


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

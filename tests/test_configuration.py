"""Cardinality ownership must not leak between causal models."""
import pytest
from autobounds import Bounder, causalProblem, DAG, Parser
from autobounds.canonicalModel import canonicalModel
from autobounds.closedProblem import closedProblem


@pytest.mark.parametrize('factory', [Bounder, causalProblem, Parser, closedProblem])
def test_model_configuration_is_independent_of_caller_and_other_models(factory):
    supplied = {'X': 3}
    first = factory(DAG('X -> Y'), supplied)
    second = factory(DAG('X -> Y'))
    assert supplied == {'X': 3}
    supplied['X'] = 4
    assert first.canModel.number_values == {'X': 3, 'Y': 2}
    assert second.canModel.number_values == {'X': 2, 'Y': 2}
    first.canModel.number_values['Y'] = 5
    assert second.canModel.number_values['Y'] == 2


def test_canonical_model_does_not_share_default_configuration():
    first, second = canonicalModel(), canonicalModel()
    first.from_dag(DAG('X -> Y'))
    first.number_values['X'] = 7
    second.from_dag(DAG('X -> Y'))
    assert second.number_values == {'X': 2, 'Y': 2}

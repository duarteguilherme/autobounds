import pandas as pd
from importlib import import_module

from autobounds.autobounds.Bounder import Bounder
from autobounds.autobounds.DAG import DAG

bounder_module = import_module("autobounds.autobounds.Bounder")


def _make_dag():
    dag = DAG()
    dag.from_structure("X -> D, X -> Y, Z -> D, D -> Y, U -> D, U -> Y", unob="U")
    return dag


def test_read_data_discrete_small_support_uses_empirical_path(monkeypatch):
    dag = _make_dag()
    bounder = Bounder(dag)

    def _fail_mnlogit(*args, **kwargs):
        raise AssertionError("MNLogit should not be called for discrete low-support covariates.")

    monkeypatch.setattr(bounder_module.sm, "MNLogit", _fail_mnlogit)

    df = pd.DataFrame(
        {
            "X": [0, 1] * 10,
            "Z": [0, 1] * 10,
            "D": [0, 1, 1, 0] * 5,
            "Y": [1, 0, 1, 0] * 5,
        }
    )
    bounder.read_data(raw=df, covariates=["X"], inference=True, categorical=False, nk=10)

    assert bounder.categorical is True
    assert bounder._used_discrete_covariate_path is True
    assert bounder._covariate_support_size == 2


def test_read_data_discrete_large_support_keeps_multinomial(monkeypatch):
    dag = _make_dag()
    bounder = Bounder(dag)
    sentinel = object()

    class _FakeMNLogit:
        def __init__(self, y, x):
            self.y = y
            self.x = x

        def fit(self):
            return sentinel

    monkeypatch.setattr(bounder_module.sm, "MNLogit", _FakeMNLogit)

    n = 30
    df = pd.DataFrame(
        {
            "X": list(range(n)),
            "Z": [0, 1] * (n // 2) + [0] * (n % 2),
            "D": [0, 1, 1] * 10,
            "Y": [1, 0, 0] * 10,
        }
    )
    bounder.read_data(raw=df, covariates=["X"], inference=True, categorical=False, nk=5)

    assert bounder.categorical is False
    assert bounder._used_discrete_covariate_path is False
    assert bounder._covariate_support_size == n
    assert bounder.main_model is sentinel

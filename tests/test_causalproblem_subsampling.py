import pandas as pd
import pytest

from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Bounder import Bounder
from autobounds.autobounds.causalProblem import causalProblem


def _fake_load_data(self, summary=None, raw=None, **kwargs):
    if raw is not None:
        df = raw if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
    elif summary is not None:
        df = summary if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
    else:
        raise ValueError("Missing data.")
    if not hasattr(self, "_test_n_rows"):
        self._test_n_rows = []
    self._test_n_rows.append(int(df.shape[0]))
    return None


def _fake_solve(self, ci=False, **kwargs):
    total_n = int(sum(getattr(self, "_test_n_rows", [])))
    return {
        "point lb dual": float(total_n),
        "point ub dual": float(total_n),
        "point lb primal": float(total_n),
        "point ub primal": float(total_n),
    }


def _fake_read_data(self, raw=None, **kwargs):
    if raw is None:
        raise ValueError("Missing raw data.")
    df = raw if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
    if not hasattr(self, "_test_n_rows"):
        self._test_n_rows = []
    self._test_n_rows.append(int(df.shape[0]))
    return None


def test_causalproblem_subsampling_ci_replays_multiple_datasets(monkeypatch):
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(Bounder, "add_prob_constraints", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 100, "Y": [0, 1] * 100}))
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 75, "Y": [1, 0] * 75}))
    problem.add_prob_constraints()

    res = problem.solve(ci=True, nsamples=8, verbose_result=False)

    # Point estimate uses full data (200 + 150).
    assert res["point lb dual"] == pytest.approx(350.0)
    # Default subsample size uses m = max(80, floor(n**0.7)) for n >= 120.
    # So each replicate uses 80 + 80 = 160 observations.
    assert res["2.5% lb bounds"] == pytest.approx(160.0)
    assert res["97.5% ub bounds"] == pytest.approx(160.0)


def test_causalproblem_subsampling_ci_uses_explicit_subsample_size(monkeypatch):
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(Bounder, "add_prob_constraints", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 80, "Y": [0, 1] * 80}))
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 60, "Y": [1, 0] * 60}))

    res = problem.solve(ci=True, nsamples=6, subsample_size=30, verbose_result=False)

    assert res["point lb dual"] == pytest.approx(280.0)
    assert res["2.5% lb bounds"] == pytest.approx(60.0)
    assert res["97.5% ub bounds"] == pytest.approx(60.0)


def test_causalproblem_subsampling_ci_supports_read_data(monkeypatch):
    monkeypatch.setattr(Bounder, "read_data", _fake_read_data)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.read_data(raw=pd.DataFrame({"D": [0, 1] * 100, "Y": [0, 1] * 100}))
    problem.read_data(raw=pd.DataFrame({"D": [0, 1] * 75, "Y": [1, 0] * 75}))

    res = problem.solve(ci=True, nsamples=8, verbose_result=False)

    assert res["point lb dual"] == pytest.approx(350.0)
    assert res["2.5% lb bounds"] == pytest.approx(160.0)
    assert res["97.5% ub bounds"] == pytest.approx(160.0)


def test_causalproblem_read_data_point_solve_does_not_use_ci_path(monkeypatch):
    monkeypatch.setattr(Bounder, "read_data", _fake_read_data)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        causalProblem,
        "_solve_with_subsampling_ci",
        lambda self, *args, **kwargs: pytest.fail("CI replay path should not run when ci=False."),
    )

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.read_data(raw=pd.DataFrame({"D": [0, 1] * 50, "Y": [0, 1] * 50}))
    out = problem.solve(ci=False, verbose_result=False)

    assert out["point lb dual"] == pytest.approx(100.0)

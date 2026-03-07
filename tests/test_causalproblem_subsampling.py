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


def _fake_read_data_noop(self, raw=None, **kwargs):
    return None


def _fake_load_data_signal(self, summary=None, raw=None, **kwargs):
    if raw is not None:
        df = raw if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
    elif summary is not None:
        df = summary if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
    else:
        raise ValueError("Missing data.")
    if not hasattr(self, "_test_signal_sum"):
        self._test_signal_sum = 0.0
        self._test_signal_n = 0
    signal = float(df["D"].mean()) if "D" in df.columns else 0.0
    self._test_signal_sum += signal * len(df)
    self._test_signal_n += len(df)
    return None


def _fake_solve_signal(self, ci=False, **kwargs):
    theta = float(self._test_signal_sum / max(1, self._test_signal_n))
    return {
        "point lb dual": theta,
        "point ub dual": theta + 1.0,
        "point lb primal": theta,
        "point ub primal": theta + 1.0,
    }


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

    res = problem.solve(
        ci=True,
        nsamples=8,
        verbose_result=False,
    )

    assert res["point lb dual"] == pytest.approx(350.0)
    assert res["ci method"] == "recentered_subsampling"
    assert pd.notna(res["2.5% lb bounds"])
    assert pd.notna(res["97.5% ub bounds"])


def test_causalproblem_subsampling_ci_replays_multiple_datasets_parallel(monkeypatch):
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(Bounder, "add_prob_constraints", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 100, "Y": [0, 1] * 100}))
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 75, "Y": [1, 0] * 75}))
    problem.add_prob_constraints()

    res = problem.solve(
        ci=True,
        nsamples=8,
        ci_workers=2,
        verbose_result=False,
    )

    assert res["point lb dual"] == pytest.approx(350.0)
    assert res["ci method"] == "recentered_subsampling"
    assert pd.notna(res["2.5% lb bounds"])
    assert pd.notna(res["97.5% ub bounds"])
    assert res["ci workers"] == 2


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
    assert res["ci method"] == "recentered_subsampling"
    assert pd.notna(res["2.5% lb bounds"])
    assert pd.notna(res["97.5% ub bounds"])


def test_causalproblem_subsampling_ci_supports_read_data(monkeypatch):
    monkeypatch.setattr(Bounder, "read_data", _fake_read_data)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 100, "Y": [0, 1] * 100})
    problem.read_data(raw=df)

    res = problem.solve(
        ci=True,
        nsamples=8,
        verbose_result=False,
    )

    assert res["point lb dual"] == pytest.approx(200.0)
    assert res["ci method"] == "recentered_subsampling"
    assert pd.notna(res["2.5% lb bounds"])
    assert pd.notna(res["97.5% ub bounds"])


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


def test_causalproblem_recentered_subsampling_ci_mode(monkeypatch):
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data_signal)
    monkeypatch.setattr(Bounder, "solve", _fake_solve_signal)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 120, "Y": [0, 1] * 120}))
    problem.load_data(raw=pd.DataFrame({"D": [0, 1, 1, 0] * 60, "Y": [1, 0, 1, 0] * 60}))

    out = problem.solve(
        ci=True,
        nsamples=20,
        ci_method="recentered_subsampling",
        verbose_result=False,
    )

    assert out["ci method"] == "recentered_subsampling"
    assert isinstance(out["2.5% lb bounds"], float)
    assert isinstance(out["97.5% ub bounds"], float)


def test_causalproblem_unsupported_ci_method_raises(monkeypatch):
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data_signal)
    monkeypatch.setattr(Bounder, "solve", _fake_solve_signal)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.load_data(raw=pd.DataFrame({"D": [0, 1] * 40, "Y": [0, 1] * 40}))

    with pytest.raises(ValueError, match="Only 'recentered_subsampling' is available"):
        problem.solve(ci=True, ci_method="empirical_subsample_quantile", verbose_result=False)


def test_causalproblem_defaults_include_k_and_covariate_flag():
    problem = causalProblem(DAG("D -> Y"))
    assert problem.K == 20
    assert problem._has_covariates is False


def test_read_data_updates_covariate_flag(monkeypatch):
    monkeypatch.setattr(Bounder, "read_data", _fake_read_data_noop)

    problem = causalProblem(DAG("D -> Y"))
    df = pd.DataFrame({"D": [0, 1], "Y": [0, 1]})

    problem.read_data(raw=df)
    assert problem._has_covariates is False

    problem.read_data(raw=df, covariates=["D"])
    assert problem._has_covariates is True


def test_no_covariate_ci_uses_subsampling_path(monkeypatch):
    monkeypatch.setattr(Bounder, "read_data", _fake_read_data_noop)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(Bounder, "solve", _fake_solve_signal)
    monkeypatch.setattr(
        causalProblem,
        "_solve_with_subsampling_ci",
        lambda self, *args, **kwargs: {"ok": True},
    )

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.read_data(raw=pd.DataFrame({"D": [0, 1], "Y": [1, 0]}))

    out = problem.solve(ci=True, verbose_result=False)
    assert out == {"ok": True}


def test_covariate_ci_uses_subsampling_path(monkeypatch):
    monkeypatch.setattr(Bounder, "read_data", _fake_read_data_noop)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(Bounder, "solve", _fake_solve_signal)
    monkeypatch.setattr(
        causalProblem,
        "_solve_with_subsampling_ci",
        lambda self, *args, **kwargs: {"cov_ci": True},
    )

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.read_data(raw=pd.DataFrame({"D": [0, 1], "Y": [0, 1]}), covariates=["D"])

    out = problem.solve(ci=True, verbose_result=False)
    assert out == {"cov_ci": True}

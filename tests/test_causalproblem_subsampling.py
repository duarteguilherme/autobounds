import pandas as pd
import pytest
from copy import deepcopy

from autobounds.DAG import DAG
from autobounds.Bounder import Bounder
from autobounds.causalProblem import causalProblem


def test_interleaved_replay_builds_identical_programs_for_each_sample(monkeypatch):
    problem = causalProblem(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'))
    problem.set_ate('X', 'Y')
    raw = pd.DataFrame({'Z': [0, 0, 1, 1] * 5,
                        'X': [0, 1, 0, 1] * 5,
                        'Y': [0, 1, 0, 1] * 5})
    problem.load_data(raw=raw)
    problem.add_constraint(problem.p('Y(X=0)=1'), '==', 0)
    problem.load_data(summary=pd.DataFrame({'X': [1, 1], 'Y': [0, 1],
                                           'prob': [0., 1.]}), do=['X'])
    problem.add_prob_constraints()
    context = problem._prepare_replay_context()
    captured = []

    def capture_program(bounder, **kwargs):
        program = bounder.write_program()
        captured.append(deepcopy((program.parameters, program.constraints,
                                  bounder.zero_parameters, bounder.unconf_first_nodes)))
        return {}

    monkeypatch.setattr(Bounder, 'solve', capture_program)
    options = dict(maxtime=5, theta=-1, verbose_optimizer=False,
                   limits=[None, None], subsample=True, subsample_rate=.7,
                   subsample_size=3, return_subsample_df=True)
    programs = []
    for seed in range(6):
        cached = problem._solve_from_operation_log(context, rep_seed=seed, **options)
        fresh = problem._solve_from_operation_log(None, rep_seed=seed, **options)
        assert captured[-2] == captured[-1]
        assert cached[1:3] == fresh[1:3]
        pd.testing.assert_frame_equal(cached[3], fresh[3])
        programs.append(captured[-1])
    assert any(program != programs[0] for program in programs[1:])
    assert all(program[2] for program in programs)  # Explicit zeros were applied.


def test_new_replay_context_picks_up_assumptions_without_changing_old_context():
    problem = causalProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'))
    raw = pd.DataFrame({'X': [0, 0, 1, 1] * 10, 'Y': [0, 1, 0, 1] * 10})
    problem.load_data(raw=raw)
    problem.set_ate('X', 'Y')
    old_context = problem._prepare_replay_context()
    problem.add_constraint(problem.p('Y(X=1)=1'), '==', .5)
    new_context = problem._prepare_replay_context()
    # Recording takes a snapshot rather than retaining the caller's data frame.
    raw['Y'] = 0
    options = dict(maxtime=5, theta=-1, verbose_optimizer=False,
                   limits=[None, None], subsample=False, subsample_rate=.7,
                   subsample_size=None, rep_seed=0)
    for context, expected in [(old_context, (-.5, .5)),
                               (new_context, (-.25, .25)),
                               (old_context, (-.5, .5))]:
        result = problem._solve_from_operation_log(context, **options)[0]
        for kind in ('primal', 'dual'):
            assert result[f'point lb {kind}'] == pytest.approx(expected[0])
            assert result[f'point ub {kind}'] == pytest.approx(expected[1])


def test_repeated_ci_solves_do_not_accumulate_constraints_or_mutate_data():
    problem = causalProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'))
    raw = pd.DataFrame({'X': [0, 1] * 20, 'Y': [0, 1] * 20})
    problem.load_data(raw=raw)
    problem.set_ate('X', 'Y')
    initial_constraints = deepcopy(problem._default_bounder.constraints)
    initial_raw = raw.copy(deep=True)
    options = dict(ci=True, nsamples=3, rep_seeds=[7, 11, 23],
                   subsample_size=10, maxtime=5, theta=-1, verbose_result=False,
                   return_subsample_dfs=True)
    first = problem.solve(**options)
    # Returned diagnostic frames are owned by the caller.
    first['subsample raw dfs'][0]['Y'] = 9
    second = problem.solve(**options)
    for key in first.keys() - {'subsample raw dfs'}:
        if isinstance(first[key], str):
            assert first[key] == second[key]
        else:
            assert first[key] == pytest.approx(second[key])
    assert second['subsample raw dfs'][0]['Y'].isin([0, 1]).all()
    assert problem._default_bounder.constraints == initial_constraints
    pd.testing.assert_frame_equal(raw, initial_raw)


@pytest.mark.parametrize('data_first', [False, True])
@pytest.mark.parametrize('stratified', [False, True])
def test_cached_replay_matches_fresh_models_with_sparse_samples(data_first, stratified):
    problem = causalProblem(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'))
    # Y=X makes these tiny subsamples compatible even when cells disappear.
    raw = pd.DataFrame({'Z': [0, 0, 1, 1] * 8,
                        'X': [0, 1, 0, 1] * 8,
                        'Y': [0, 1, 0, 1] * 8})
    if not data_first:
        problem.set_ate('X', 'Y')
    if stratified:
        raw['C'] = [0] * 16 + [1] * 16
        problem.read_data(raw=raw, covariates=['C'])
    else:
        problem.load_data(raw=raw)
    if data_first:
        problem.set_ate('X', 'Y')
    # An explicit zero-probability cell must be reapplied in the original order.
    problem.load_data(summary=pd.DataFrame({'X': [0, 0], 'Y': [0, 1],
                                           'prob': [1., 0.]}), do=['X'])
    problem.add_prob_constraints()
    context = problem._prepare_replay_context()
    assert context['fast_path']
    prototype = context['prototype']
    assert prototype.Parser is not problem._default_bounder.Parser
    state = deepcopy((prototype.constraints, prototype.zero_parameters,
                      prototype.unconf_first_nodes))
    # Disable width-based early stopping when comparing primal and dual optima.
    options = dict(maxtime=5, theta=-1, verbose_optimizer=False,
                   limits=[None, None], subsample_rate=.7,
                   subsample_size=4, return_subsample_df=True)
    for subsample, seed in [(False, 0), (True, 0), (True, 1), (True, 0)]:
        cached = problem._solve_from_operation_log(
            context, subsample=subsample, rep_seed=seed, **options)
        fresh = problem._solve_from_operation_log(
            None, subsample=subsample, rep_seed=seed, **options)
        for key in ('point lb dual', 'point ub dual',
                    'point lb primal', 'point ub primal'):
            assert cached[0][key] == pytest.approx(fresh[0][key], abs=1e-8)
        assert cached[1:3] == fresh[1:3]
        if subsample:
            pd.testing.assert_frame_equal(cached[3], fresh[3])
        assert (prototype.constraints, prototype.zero_parameters,
                prototype.unconf_first_nodes) == state


def test_replay_copies_share_structure_but_isolate_constraints():
    prototype = Bounder(DAG('X -> Y, U -> X, U -> Y', unob='U'))
    prototype.set_ate('X', 'Y')
    first, second = prototype._copy_for_replay(), prototype._copy_for_replay()
    assert first.Parser is second.Parser is prototype.Parser
    assert first.canModel is second.canModel is prototype.canModel
    first.constraints[0][0][1].append('mutation')
    first.zero_parameters.add('X0.Y00')
    first.unconf_first_nodes.append(('X0', .25))
    assert second.constraints == prototype.constraints
    assert second.zero_parameters == prototype.zero_parameters == set()
    assert second.unconf_first_nodes == prototype.unconf_first_nodes == []


def test_cached_ci_matches_serial_and_parallel_workers():
    problem = causalProblem(DAG('X -> Y, U -> X, U -> Y', unob='U'))
    problem.load_data(raw=pd.DataFrame({'X': [0, 1] * 20, 'Y': [0, 1] * 20}))
    problem.set_ate('X', 'Y')
    options = dict(ci=True, nsamples=4, rep_seeds=[0, 1, 2, 3],
                   subsample_size=8, maxtime=5, theta=-1, verbose_result=False)
    serial = problem.solve(ci_workers=1, **options)
    parallel = problem.solve(ci_workers=2, **options)
    for key, value in serial.items():
        if key == 'ci workers':
            continue
        if isinstance(value, str):
            assert parallel[key] == value
        else:
            assert parallel[key] == pytest.approx(value, abs=1e-8)


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
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data)
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
    monkeypatch.setattr(Bounder, "load_data", _fake_load_data)
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
    problem = causalProblem(DAG("D -> Y"))
    df = pd.DataFrame({"D": [0, 1], "Y": [0, 1]})

    problem.read_data(raw=df)
    assert problem._has_covariates is False

    problem.read_data(raw=df, covariates=["D"])
    assert problem._has_covariates is True


def test_no_covariate_ci_uses_subsampling_path(monkeypatch):
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


def test_covariate_ci_uses_strata_load_data_without_bounder_covariates(monkeypatch):
    read_covariates_args = []

    def _fake_load_data_track(self, summary=None, raw=None, covariates=None, **kwargs):
        read_covariates_args.append(covariates)
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

    monkeypatch.setattr(Bounder, "load_data", _fake_load_data_track)
    monkeypatch.setattr(Bounder, "solve", _fake_solve)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    df = pd.DataFrame(
        {
            "X": [0] * 90 + [1] * 30,
            "D": [0, 1] * 60,
            "Y": [1, 0] * 60,
        }
    )
    problem.read_data(raw=df, covariates=["X"])

    out = problem.solve(ci=True, nsamples=6, ci_workers=2, verbose_result=False)

    assert out["ci method"] == "recentered_subsampling"
    assert len(read_covariates_args) > 0
    assert all(c is None for c in read_covariates_args[1:])


def test_stratified_subsample_preserves_covariate_support():
    problem = causalProblem(DAG("X -> Y"))
    df = pd.DataFrame(
        {
            "X": [0] * 90 + [1] * 30,
            "Y": [0, 1] * 60,
        }
    )
    sub = problem._subsample_rows_stratified(
        df,
        covariates=["X"],
        subsample_rate=2.0 / 3.0,
        subsample_size=40,
        random_state=123,
    )

    assert len(sub) == 40
    levels = set(sub["X"].unique().tolist())
    assert levels == {0, 1}


def test_solve_discrete_outcome_thresholds_replays_binary_outcomes(monkeypatch):
    seen_outcome_supports = []

    def _fake_load_data_threshold(self, summary=None, raw=None, **kwargs):
        if raw is not None:
            df = raw if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
        elif summary is not None:
            df = summary if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
        else:
            raise ValueError("Missing data.")
        seen_outcome_supports.append(tuple(sorted(pd.Series(df["Y"]).dropna().unique().tolist())))
        return None

    monkeypatch.setattr(Bounder, "load_data", _fake_load_data_threshold)
    monkeypatch.setattr(Bounder, "set_ate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        Bounder,
        "solve",
        lambda self, **kwargs: {
            "point lb dual": 0.0,
            "point ub dual": 1.0,
            "point lb primal": 0.0,
            "point ub primal": 1.0,
        },
    )

    problem = causalProblem(DAG("D -> Y"))
    problem.set_ate("D", "Y")
    problem.load_data(
        raw=pd.DataFrame(
            {
                "D": [0, 1] * 6,
                "Y": [0, 1, 2, 3] * 3,
            }
        )
    )

    out = problem.solve_discrete_outcome_thresholds("D", "Y", direction="geq", verbose_result=False)

    assert out["thresholds"] == [1, 2, 3]
    assert seen_outcome_supports[0] == (0, 1, 2, 3)
    assert set(seen_outcome_supports[1:]) == {(0, 1)}

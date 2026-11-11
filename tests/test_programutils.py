import math
import pytest

from autobounds.autobounds import ProgramUtils as pu


class _FakeProcess:
    def __init__(self):
        self.terminated = 0
        self.alive = True
        self.exitcode = None
        self.sentinel = object()

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated += 1


def _write_final_log(path, bound):
    path.write_text(
        "SCIP Status : problem is solved [optimal solution found]\n"
        f"Primal Bound : {bound}\nDual Bound : {bound}\nSolving Time : 0.01\n"
    )


def test_monitor_preserves_primal_and_dual_bounds_when_stopping_early(monkeypatch):
    lower, upper = _FakeProcess(), _FakeProcess()
    rows = {'lower': {'primal': .402, 'dual': .4, 'time': .1},
            'upper': {'primal': .403, 'dual': .405, 'time': .1}}
    monkeypatch.setattr(pu, 'parse_particular_bound_scip',
                        lambda filename, count: (1, [rows[filename].copy()]))
    monkeypatch.setattr(pu, 'check_process_end_scip', lambda *args: -1)
    monkeypatch.setattr(pu, 'wait_for_processes',
                        lambda *args, **kwargs: pytest.fail('Stopping threshold was ignored'))
    result = pu.parse_bounds_scip(lower, upper, filelower='lower', fileupper='upper',
                                 theta=.01, verbose=False)
    for index, name in enumerate(('lower', 'upper')):
        assert result[index]['primal'] == rows[name]['primal']
        assert result[index]['dual'] == rows[name]['dual']
        assert result[index]['end'] == -1
    assert result[2] == pytest.approx(.005)
    assert lower.terminated == upper.terminated == 1


def test_zero_deadline_does_not_wait_for_workers(monkeypatch, tmp_path):
    lower, upper = _FakeProcess(), _FakeProcess()
    log = tmp_path / 'empty.log'
    log.touch()
    monkeypatch.setattr(pu, 'wait_for_processes',
                        lambda *args, **kwargs: pytest.fail('Expired deadline was ignored'))
    result = pu.parse_bounds_scip(lower, upper, filelower=log, fileupper=log,
                                 maxtime=0, verbose=False)
    assert lower.terminated == upper.terminated == 1
    assert result[0]['end'] == result[1]['end'] == -1


def test_monitor_wakes_on_worker_completion_and_preserves_bounds(monkeypatch, tmp_path):
    lower, upper = _FakeProcess(), _FakeProcess()
    lower_log, upper_log = tmp_path / "lower.log", tmp_path / "upper.log"
    lower_log.touch()
    upper_log.touch()
    # One worker has already finished; waiting on it would spin instead of
    # blocking for the remaining worker.
    _write_final_log(lower_log, -0.5)
    lower.alive, lower.exitcode = False, 0
    waits = []

    def finish_upper(handles, timeout):
        assert handles == [upper.sentinel]
        waits.append(timeout)
        _write_final_log(upper_log, 0.5)
        upper.alive, upper.exitcode = False, 0
        return handles

    def reject_sleep(*args):
        pytest.fail("Finished SCIP results must not be delayed by a fixed sleep")

    monkeypatch.setattr(pu, "wait_for_processes", finish_upper)
    monkeypatch.setattr(pu.time, "sleep", reject_sleep)
    result = pu.parse_bounds_scip(lower, upper, filelower=lower_log,
                                 fileupper=upper_log, verbose=False)
    assert len(waits) == 1
    assert result[0]["primal"] == result[0]["dual"] == -0.5
    assert result[1]["primal"] == result[1]["dual"] == 0.5
    assert result[0]["end"] == result[1]["end"] == 1
    assert lower.terminated == upper.terminated == 0


def test_monitor_limits_wait_to_remaining_deadline(monkeypatch, tmp_path):
    lower, upper = _FakeProcess(), _FakeProcess()
    log = tmp_path / "empty.log"
    log.touch()
    clock = [0.0]
    waits = []

    def wait_until_deadline(handles, timeout):
        waits.append(timeout)
        clock[0] += timeout
        return []

    monkeypatch.setattr(pu.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pu, "wait_for_processes", wait_until_deadline)
    result = pu.parse_bounds_scip(lower, upper, filelower=log, fileupper=log,
                                 maxtime=0.025, verbose=False)
    assert waits == [0.025]
    assert lower.terminated == upper.terminated == 1
    assert result[0]["end"] == result[1]["end"] == -1


def test_final_status_waits_for_worker_to_finish_writing(monkeypatch, tmp_path):
    worker = _FakeProcess()
    log = tmp_path / "partial.log"
    log.write_text("SCIP Status : problem is solved [optimal solution found]\n")
    assert pu.check_process_end_scip(worker, log, False) == -1
    assert worker.terminated == 0
    _write_final_log(log, 0.25)
    worker.alive, worker.exitcode = False, 0
    assert pu.check_process_end_scip(worker, log, False) == 1
    assert worker.terminated == 0


@pytest.mark.parametrize("exitcode", [0, 1])
def test_exited_worker_without_results_does_not_poll_forever(tmp_path, exitcode):
    worker = _FakeProcess()
    worker.alive, worker.exitcode = False, exitcode
    log = tmp_path / "failed.log"
    log.touch()
    with pytest.raises(RuntimeError, match="SCIP worker exited"):
        pu.check_process_end_scip(worker, log, False)


def test_parse_bounds_scip_handles_partial_completion(monkeypatch):
    monkeypatch.setattr(pu.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pu,
        "parse_particular_bound_scip",
        lambda _filename, n_bound: (n_bound, []),
    )
    monkeypatch.setattr(
        pu,
        "check_process_end_scip",
        lambda _proc, filename, _verbose=True: 1 if filename == "lower.log" else 0,
    )
    monkeypatch.setattr(
        pu,
        "get_final_bound_scip",
        lambda _filename: {"primal": -0.25, "dual": -0.2, "time": 0.1},
    )

    p_lower = _FakeProcess()
    p_upper = _FakeProcess()
    i, j, _theta, _epsilon = pu.parse_bounds_scip(
        p_lower,
        p_upper,
        filelower="lower.log",
        fileupper="upper.log",
        verbose=False,
    )

    assert i["end"] == 1
    assert j["end"] == 0
    assert math.isnan(j["primal"])
    assert math.isnan(j["dual"])
    assert math.isnan(j["time"])


def test_parse_bounds_scip_infeasible_writes_safe_output(monkeypatch, tmp_path):
    monkeypatch.setattr(pu.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pu,
        "parse_particular_bound_scip",
        lambda _filename, n_bound: (n_bound, []),
    )
    monkeypatch.setattr(
        pu,
        "check_process_end_scip",
        lambda _proc, _filename, _verbose=True: 0,
    )

    output_file = tmp_path / "bounds.csv"
    p_lower = _FakeProcess()
    p_upper = _FakeProcess()
    i, j, _theta, _epsilon = pu.parse_bounds_scip(
        p_lower,
        p_upper,
        filelower="lower.log",
        fileupper="upper.log",
        output=str(output_file),
        verbose=False,
    )

    assert i["end"] == 0
    assert j["end"] == 0
    lines = output_file.read_text().strip().splitlines()
    assert lines[0] == "bound,primal,dual,time"
    assert lines[1].startswith("lb,")
    assert lines[2].startswith("ub,")
    assert "nan" in lines[1]
    assert "nan" in lines[2]

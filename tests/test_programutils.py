import math

from autobounds.autobounds import ProgramUtils as pu


class _FakeProcess:
    def __init__(self):
        self.terminated = 0

    def terminate(self):
        self.terminated += 1


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

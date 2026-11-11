"""Historical helpers must fail explicitly rather than block or return None."""
import pytest
from autobounds.closedProblem import get_subtraction_bounds
from autobounds.utils import parse_estimand, introduce_prob_into_progr
from autobounds.canonicalModel import canonicalModel
from autobounds.ProgramUtils import get_final_bound


@pytest.mark.parametrize('function,args', [
    (get_subtraction_bounds, (None,) * 6),
    (parse_estimand, (None, None)),
    (introduce_prob_into_progr, (None, None)),
    (canonicalModel().find_index, (None, None)),
])
def test_unimplemented_legacy_helpers_fail_explicitly(function, args):
    with pytest.raises(NotImplementedError):
        function(*args)


def test_legacy_log_reader_never_prompts_for_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def unexpected_input(*args):
        pytest.fail('Log reading must not prompt for input')
    monkeypatch.setattr('builtins.input', unexpected_input)
    (tmp_path / '.lower.log').write_text(
        'Upper bound: 0.4\nLower bound: 0.3\nTotal solve time: 1.2s\n')
    assert get_final_bound('.lower.log') == {'primal': 0.4, 'dual': 0.3, 'time': 1.2}

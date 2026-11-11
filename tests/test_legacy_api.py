"""Historical helpers must fail explicitly rather than block or return None."""
import pytest
from autobounds.closedProblem import get_subtraction_bounds
from autobounds.utils import parse_estimand, introduce_prob_into_progr
from autobounds.canonicalModel import canonicalModel


@pytest.mark.parametrize('function,args', [
    (get_subtraction_bounds, (None,) * 6),
    (parse_estimand, (None, None)),
    (introduce_prob_into_progr, (None, None)),
    (canonicalModel().find_index, (None, None)),
])
def test_unimplemented_legacy_helpers_fail_explicitly(function, args):
    with pytest.raises(NotImplementedError):
        function(*args)

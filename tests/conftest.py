"""Feature tests skip only dependencies that are absent from an installation."""
import pytest


@pytest.fixture
def inference_extra():
    pytest.importorskip('scipy')
    pytest.importorskip('statsmodels.api')

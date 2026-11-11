"""Feature dependencies must fail helpfully without breaking numerical use."""
import pytest

from autobounds import DAG, causalProblem
from autobounds import optional
from autobounds.data_utils import solve_gaussian, solve_kl_p
from autobounds.ProgramUtils import plot_bounds


@pytest.mark.parametrize('module,extra,operation', [
    ('matplotlib', 'plot', lambda: DAG('X -> Y').plot()),
    ('plotnine', 'plot', lambda: plot_bounds(None)),
    ('scipy', 'inference', lambda: solve_gaussian(100, [.5, .5], .05)),
    ('scipy', 'inference', lambda: solve_kl_p(100, 2, .5, .05)),
])
def test_missing_feature_dependency_explains_installation(monkeypatch, module, extra, operation):
    original = optional.import_module

    def missing(name):
        if name.split('.')[0] == module:
            raise ModuleNotFoundError(f'No module named {module}', name=module)
        return original(name)

    monkeypatch.setattr(optional, 'import_module', missing)
    with pytest.raises(ImportError, match=rf'autobounds\[{extra}\]'):
        operation()


def test_regression_covariate_path_explains_missing_inference_extra(monkeypatch):
    import pandas as pd
    original = optional.import_module

    def missing(name):
        if name.startswith('statsmodels'):
            raise ModuleNotFoundError('No statsmodels', name='statsmodels')
        return original(name)

    monkeypatch.setattr(optional, 'import_module', missing)
    problem = causalProblem(DAG('X -> D, D -> Y, X -> Y'))
    data = pd.DataFrame({'X': range(10), 'D': [0, 1] * 5, 'Y': [1, 0] * 5})
    with pytest.raises(ImportError, match=r'autobounds\[inference\]'):
        problem.read_data(data, covariates=['X'], categorical=False, nk=1)


def test_dependency_internal_import_failures_are_not_masked(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError('broken internal installation', name='internal_dependency')

    monkeypatch.setattr(optional, 'import_module', broken)
    with pytest.raises(ModuleNotFoundError, match='broken internal installation'):
        optional.require_optional('statsmodels.api', 'inference')

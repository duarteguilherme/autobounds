"""Fresh-interpreter import contracts for portable solver startup."""
import subprocess
import sys
import textwrap
import pytest


def run_fresh(source):
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_imports_only_numerical_solver_dependencies():
    run_fresh('''
        import sys
        import autobounds.backends.scip
        from autobounds.model import MathematicalModel
        from autobounds import Program
        forbidden = {'pandas', 'scipy', 'sympy', 'matplotlib', 'plotnine', 'statsmodels'}
        assert forbidden.isdisjoint(sys.modules), forbidden.intersection(sys.modules)
    ''')


def test_causal_declarations_do_not_import_plotting_or_statistical_models():
    run_fresh('''
        import sys
        from autobounds import DAG, causalProblem
        problem = causalProblem(DAG('X -> Y'))
        problem.set_ate('X', 'Y')
        problem.compile()
        forbidden = {'scipy', 'sympy', 'matplotlib', 'plotnine', 'statsmodels'}
        assert forbidden.isdisjoint(sys.modules), forbidden.intersection(sys.modules)
    ''')


def test_public_class_exports_survive_submodule_imports_and_pickle():
    run_fresh('''
        import importlib
        import pickle
        import autobounds
        for name, (module_name, attribute) in autobounds._EXPORTS.items():
            module = importlib.import_module('autobounds.' + module_name)
            expected = getattr(module, attribute)
            assert getattr(autobounds, name) is expected, name
            assert pickle.loads(pickle.dumps(expected)) is expected, name
        namespace = {}
        exec('from autobounds import *', namespace)
        for name in autobounds.__all__:
            assert namespace[name] is getattr(autobounds, name), name
        assert set(autobounds.__all__) <= set(dir(autobounds))
    ''')


def test_plotting_imports_are_available_when_requested():
    pytest.importorskip("plotnine")
    run_fresh('''
        import sys
        from unittest.mock import patch
        import pandas as pd
        from autobounds import DAG
        from autobounds.ProgramUtils import plot_bounds
        import matplotlib
        matplotlib.use('Agg')
        with patch('matplotlib.pyplot.show'):
            DAG('X -> Y').plot()
        assert 'matplotlib.pyplot' in sys.modules
        # Plot construction also verifies that the deferred imports are complete.
        frame = pd.DataFrame({'seconds': [0, 1], 'lb.dual': [-1, -.5],
                              'ub.dual': [1, .5], 'lb.prim': [0, 0],
                              'ub.prim': [.25, .25]})
        plot_bounds(frame).draw()
        assert 'plotnine' in sys.modules
    ''')

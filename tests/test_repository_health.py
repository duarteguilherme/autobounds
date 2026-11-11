"""Guard against broken package sources and silently overwritten tests."""
import ast
from pathlib import Path
import subprocess
import sys

import autobounds
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_package_sources_compile():
    for path in Path(autobounds.__file__).parent.rglob('*.py'):
        compile(path.read_bytes(), str(path), 'exec')


def test_test_names_are_unique_within_each_scope():
    for path in Path(__file__).parent.glob('test_*.py'):
        tree = ast.parse(path.read_bytes())
        for scope in ast.walk(tree):
            if isinstance(scope, (ast.Module, ast.ClassDef)):
                names = [node.name for node in scope.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name.startswith('test_')]
                assert len(names) == len(set(names)), str(path)


@pytest.mark.parametrize('name', [
    'paper_iv_joint_ci_compare', 'paper_iv_ate_joint_ci_compare',
])
@pytest.mark.parametrize('as_module', [False, True])
def test_paper_example_entrypoints(name, as_module):
    arguments = (['-m', f'examples.{name}'] if as_module
                 else [str(ROOT / 'examples' / f'{name}.py')])
    result = subprocess.run(
        [sys.executable, *arguments, '--help'], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout

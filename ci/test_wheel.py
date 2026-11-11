"""Build, install, and test a wheel without importing the source checkout.

Run with the Python version to test; requires the build frontend. Network
access is needed to resolve dependencies into the isolated build/runtime envs.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


def run(command, *, cwd, env):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbolic', action='store_true')
    parser.add_argument('--extras', default='', help='Comma-separated plot,inference,symbolic extras')
    args = parser.parse_args()
    extras = set(filter(None, (item.strip() for item in args.extras.split(','))))
    if args.symbolic:
        extras.add('symbolic')
    if extras - {'plot', 'inference', 'symbolic'}:
        parser.error('Supported extras: plot, inference, symbolic')
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['MPLBACKEND'] = 'Agg'
    with tempfile.TemporaryDirectory(prefix='autobounds-wheel-test-') as directory:
        # Resolve macOS /var aliases and Windows short names before comparison.
        temporary = Path(directory).resolve()
        env['MPLCONFIGDIR'] = str(temporary / 'matplotlib')
        distributions = temporary / 'dist'
        # build creates an sdist, then builds the wheel from that sdist.
        run([sys.executable, '-m', 'build', '--outdir', distributions], cwd=root, env=env)
        wheels = list(distributions.glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError(f'Expected one built wheel, found {wheels}')
        runtime = temporary / 'runtime'
        venv.EnvBuilder(with_pip=True).create(runtime)
        python = runtime / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        package = str(wheels[0]) + ('[' + ','.join(sorted(extras)) + ']' if extras else '')
        run([python, '-m', 'pip', 'install', package, 'pytest'], cwd=temporary, env=env)
        run([python, '-m', 'pip', 'check'], cwd=temporary, env=env)
        run([python, '-m', 'pip', 'freeze'], cwd=temporary, env=env)
        suite = temporary / 'suite'
        suite.mkdir()
        shutil.copytree(root / 'tests', suite / 'tests',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(root / 'examples', suite / 'examples',
                        ignore=lambda folder, names: [
                            name for name in names if not name.endswith('.py')])
        shutil.copy2(root / 'pyproject.toml', suite / 'pyproject.toml')
        check = (
            'import sys; from pathlib import Path; import autobounds; '
            'location = Path(autobounds.__file__).resolve(); '
            'print(sys.version); print("Installed package:", location); '
            f'assert location.is_relative_to(Path({str(runtime)!r})), location; '
            f'assert not location.is_relative_to(Path({str(root)!r})), location'
        )
        check += '; import importlib.util'
        if 'symbolic' in extras:
            check += '; import cdd.gmp; print("Exact cdd backend available")'
        else:
            check += '; assert importlib.util.find_spec("cdd") is None'
        if 'plot' in extras:
            check += '; import matplotlib, plotnine'
        else:
            check += '; assert importlib.util.find_spec("plotnine") is None'
        if 'inference' in extras:
            check += '; import scipy, statsmodels.api'
        if not (extras & {'plot', 'inference'}):
            check += '; assert all(importlib.util.find_spec(m) is None for m in ' + repr(
                ['matplotlib', 'scipy', 'statsmodels', 'tqdm']) + ')'
        run([python, '-c', check], cwd=suite, env=env)
        run([python, '-m', 'pytest', '-ra', '--tb=short'], cwd=suite, env=env)


if __name__ == '__main__':
    main()

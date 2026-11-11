"""The public compatibility fields must describe the canonical release."""
import subprocess
import sys

import autobounds


def test_version_fields_preserve_prerelease():
    info = autobounds.version_info
    suffix = {"final": "", "alpha": "a", "beta": "b", "candidate": "rc"}[
        info.releaselevel]
    expected = f"{info.major}.{info.minor}.{info.micro}"
    if suffix:
        expected += f"{suffix}{info.serial}"
    assert autobounds.__version__ == expected
    assert autobounds.version.startswith(expected + " (")


def test_version_import_keeps_solver_workers_lightweight():
    subprocess.run([
        sys.executable, "-c",
        "import sys; from autobounds import __version__, version_info; "
        "assert not {'numpy', 'pandas', 'sympy', 'pyscipopt'} & sys.modules.keys()",
    ], check=True)

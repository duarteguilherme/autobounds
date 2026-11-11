"""Canonical release version, readable by setuptools without package imports."""
from collections import namedtuple
import platform
import re

__version__ = "0.1.0rc1"

VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")
_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?", __version__)
if _match is None:
    raise ValueError(f"Unsupported release version: {__version__}")
version_info = VersionInfo(
    *map(int, _match.group(1, 2, 3)),
    {None: "final", "a": "alpha", "b": "beta", "rc": "candidate"}[_match[4]],
    int(_match[5] or 0),
)
version = (
    f"{__version__} "
    f"({platform.python_implementation()} {platform.python_version()})"
)

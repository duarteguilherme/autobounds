from .DAG import DAG
from .causalProblem import causalProblem, respect_to
from .bounder import Bounder
from .Parser import Parser
from .Program import Program
from .Q import Q, Query
from collections import namedtuple
import platform


VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial") # machine readable version info
version_info = VersionInfo(0, 0, 2, "alpha", 0)
version = f"{version_info.major}.{version_info.minor}.{version_info.micro} ({platform.python_implementation()} {platform.python_version()})"
__version__ = f"{version_info.major}.{version_info.minor}.{version_info.micro}"


__all__ = [ 
    "version",
    "version_info",
    "__version__",
           "DAG", "Bounder", "causalProblem", "Parser", "Program", "Query", "Q", "respect_to" ]

"""Public interfaces, loaded on demand so solver workers stay lightweight."""
from importlib import import_module
import sys
from types import ModuleType

_EXPORTS = {
    'DAG': ('DAG', 'DAG'),
    'causalProblem': ('causalProblem', 'causalProblem'),
    'respect_to': ('causalProblem', 'respect_to'),
    'Bounder': ('bounder', 'Bounder'),
    'Parser': ('Parser', 'Parser'),
    'Program': ('Program', 'Program'),
    'Q': ('Q', 'Q'),
    'Query': ('Q', 'Query'),
    'SymbolicProblem': ('SymbolicProblem', 'SymbolicProblem'),
    'LinearResponseModel': ('linear_response', 'LinearResponseModel'),
    'SymbolicBounds': ('symbolic_results', 'SymbolicBounds'),
    **{name: ('model', name) for name in
       ('MathematicalModel', 'Variable', 'Term', 'Polynomial', 'Constraint')},
    **{name: ('results', name) for name in ('EndpointResult', 'SolverResult')},
}


class _PublicInterfaces(ModuleType):
    def __getattribute__(self, name):
        # Import machinery sets package.Program to the submodule. Resolve public
        # names even then, preserving the original class exports in any order.
        if name in _EXPORTS:
            module, attribute = _EXPORTS[name]
            return getattr(import_module('.' + module, __name__), attribute)
        return super().__getattribute__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(_EXPORTS))


sys.modules[__name__].__class__ = _PublicInterfaces

from ._version import VersionInfo, __version__, version, version_info


__all__ = [ 
    "version",
    "version_info",
    "__version__",
           "DAG", "Bounder", "causalProblem", "Parser", "Program", "Query", "Q", "respect_to",
           "SymbolicProblem", "LinearResponseModel", "SymbolicBounds",
           "MathematicalModel", "Variable", "Term", "Polynomial", "Constraint",
           "EndpointResult", "SolverResult" ]

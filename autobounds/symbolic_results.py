"""Exact symbolic envelopes and their numerical evaluation."""
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import numpy as np
import sympy as sp

@dataclass(frozen=True)
class SymbolicBounds:
    """Complete sharp envelopes on the model's observable compatibility region.

    Certificate entries multiply (1, p0, ..., pr). All coefficients and
    response-wise inequalities are exact. A complete envelope can still
    contain expressions redundant on the observable compatibility region.
    Compatibility equalities and target-free inequalities
    together with lower <= upper characterize the observable region.
    """

    lower_expressions: tuple
    upper_expressions: tuple
    symbols: tuple
    lower_certificates: tuple
    upper_certificates: tuple
    compatibility_equalities: tuple = ()
    sharp: bool = True
    complete: bool = True
    method: str = "exact_cdd_hull"
    compatibility_inequalities: tuple = ()

    @cached_property
    def lower(self):
        return sp.Max(*self.lower_expressions)

    @cached_property
    def upper(self):
        return sp.Min(*self.upper_expressions)

    @cached_property
    def _evaluation_symbols(self):
        return tuple(sorted(self.lower.free_symbols | self.upper.free_symbols,
                            key=sp.default_sort_key))

    @cached_property
    def _numeric_envelopes(self):
        """Compile the simplified affine envelopes without altering exact data."""
        lower = self.lower.args if isinstance(self.lower, sp.Max) else (self.lower,)
        upper = self.upper.args if isinstance(self.upper, sp.Min) else (self.upper,)
        try:
            matrix, rhs = sp.linear_eq_to_matrix(
                lower + upper, self._evaluation_symbols)
            coefficients = np.asarray(matrix, dtype=float)
            constants = -np.asarray(rhs, dtype=float).ravel()
        except (ValueError, TypeError, OverflowError):
            # Preserve symbolic substitution for non-affine or unrepresentable
            # expressions in manually constructed SymbolicBounds objects.
            return None
        if not (np.isfinite(coefficients).all() and np.isfinite(constants).all()):
            return None
        coefficients.setflags(write=False)
        constants.setflags(write=False)
        return coefficients, constants, len(lower)

    def evaluate(self, probabilities):
        """Evaluate the formulas, assuming compatible population probabilities.

        Floating-point inputs use cached affine coefficients. Exact inputs
        retain symbolic substitution before conversion of the bounds to floats.
        This does not test membership in the observable compatibility polytope.
        Keys may be symbols or their string labels.
        """
        values = {str(key): value for key, value in probabilities.items()}
        required = self._evaluation_symbols
        missing = [str(symbol) for symbol in required if str(symbol) not in values]
        if missing:
            raise ValueError(f"Missing probabilities: {sorted(missing)}")
        substitutions = {symbol: values[str(symbol)] for symbol in required}
        arguments = tuple(substitutions.values())
        if (any(isinstance(value, (float, np.floating)) for value in arguments)
                and all(isinstance(value, (int, float, np.integer, np.floating))
                        for value in arguments)):
            compiled = self._numeric_envelopes
            if compiled is not None:
                coefficients, constants, split = compiled
                try:
                    with np.errstate(over='ignore', invalid='ignore'):
                        evaluated = (coefficients @ np.asarray(arguments, dtype=float)
                                     + constants)
                    if np.isfinite(evaluated).all():
                        return (float(evaluated[:split].max()),
                                float(evaluated[split:].min()))
                except OverflowError:
                    pass  # Large exact integers can still cancel symbolically.
        return (float(self.lower.subs(substitutions)),
                float(self.upper.subs(substitutions)))

    def __str__(self):
        return f"{self.lower} <= estimand <= {self.upper}"

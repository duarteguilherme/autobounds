"""Exact symbolic LP bounds for finite, unrestricted response distributions."""

from dataclasses import dataclass
from fractions import Fraction

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

    @property
    def lower(self):
        return sp.Max(*self.lower_expressions)

    @property
    def upper(self):
        return sp.Min(*self.upper_expressions)

    def evaluate(self, probabilities):
        """Evaluate the formulas, assuming compatible population probabilities.

        This is substitution, not a test of membership in the observable
        compatibility polytope. Keys may be symbols or their string labels.
        """
        values = {str(key): value for key, value in probabilities.items()}
        required = self.lower.free_symbols | self.upper.free_symbols
        missing = [str(symbol) for symbol in required if str(symbol) not in values]
        if missing:
            raise ValueError(f"Missing probabilities: {sorted(missing)}")
        substitutions = {symbol: values[str(symbol)] for symbol in required}
        return (float(self.lower.subs(substitutions)),
                float(self.upper.subs(substitutions)))

    def __str__(self):
        return f"{self.lower} <= estimand <= {self.upper}"


class LinearResponseModel:
    """The exact model q >= 0, sum(q) = 1, M q = p, target c' q.

    Rows of ``matrix`` are observable event indicators. ``target`` accepts
    signed rational coefficients. Every distribution over the supplied
    response types must be admissible apart from these linear equations.
    No independence constraints on q are implicit.
    """

    def __init__(self, matrix, target, labels=None, response_types=None):
        self.target = tuple(sp.Rational(value) for value in target)
        if not self.target:
            raise ValueError("At least one admissible response type is required.")
        self.matrix = tuple(tuple(sp.Rational(value) for value in row) for row in matrix)
        if any(len(row) != len(self.target) for row in self.matrix):
            raise ValueError("Every incidence row must match the target length.")
        if any(value not in (0, 1) for row in self.matrix for value in row):
            raise ValueError("Data rows must be event indicators (zero or one).")
        if labels is None:
            labels = [f"p{i}" for i in range(len(self.matrix))]
        self.labels = tuple(map(str, labels))
        if len(self.labels) != len(self.matrix) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Provide one distinct label per data row.")
        self.symbols = tuple(sp.Symbol(label, real=True) for label in self.labels)
        self.response_types = tuple(range(len(self.target)) if response_types is None
                                    else response_types)
        if len(self.response_types) != len(self.target):
            raise ValueError("Response type labels must match the target length.")

    def solve(self):
        """Return complete sharp bounds using the exact GMP/cdd hull solver.

        Install ``autobounds[symbolic]`` to use this solver. Dependent
        observable coordinates are removed exactly before hull computation.

        Normalized nonnegative mixtures of response points are exactly the
        feasible (p, theta) pairs. Thus every hull inequality involving theta
        supplies an endpoint expression, and their envelopes are sharp.
        If c is in the row span of [1; M], theta is identified and recovered
        directly from that affine relation instead of treating it as an
        independent hull coordinate.
        """
        try:
            import cdd.gmp as cdd
        except ImportError as exc:
            raise ImportError(
                "The exact cdd backend requires pycddlib>=3.0.2,<4. "
                "Install it with: python -m pip install 'autobounds[symbolic]'. "
            ) from exc

        full = sp.Matrix([(1,) * len(self.target)] + list(self.matrix))
        augmented = full.col_join(sp.Matrix([self.target]))
        pivots = augmented.T.rref()[1]
        target_independent = full.rows in pivots
        independent = tuple(i for i in pivots if i < full.rows)
        reduced = full[list(independent), :]
        rhs = (sp.Integer(1),) + self.symbols
        equalities = tuple(sp.Eq(sum(v * b for v, b in zip(vector, rhs)), 0)
                           for vector in full.T.nullspace())

        def lift(values):
            certificate = [sp.Integer(0)] * full.rows
            for index, value in zip(independent, values):
                certificate[index] = value
            return tuple(certificate)

        lower, upper, compatibility = set(), set(), set()
        if not target_independent:
            coefficients, _ = reduced.T.gauss_jordan_solve(sp.Matrix(self.target))
            certificate = lift(coefficients)
            lower.add(certificate)
            upper.add(certificate)

        coordinates = reduced
        if target_independent:
            coordinates = coordinates.col_join(sp.Matrix([self.target]))
        # A singleton in zero-dimensional affine space needs no conversion.
        if coordinates.rows > 1:
            points = sorted({tuple(Fraction(int(v.p), int(v.q))
                                   for v in coordinates[:, s])
                             for s in range(coordinates.cols)})
            generators = cdd.matrix_from_array(points, rep_type=cdd.RepType.GENERATOR)
            polyhedron = cdd.polyhedron_from_matrix(generators)
            inequalities = cdd.copy_inequalities(polyhedron)
            if inequalities.lin_set:
                raise RuntimeError("Unexpected equality after exact affine dimension reduction.")
            for row in inequalities.array:
                row = tuple(sp.Rational(value.numerator, value.denominator) for value in row)
                # cdd's convention is a0 + a'p + b*theta >= 0.
                coefficient = row[-1] if target_independent else sp.Integer(0)
                observed = row[:-1] if target_independent else row
                if coefficient:
                    certificate = lift(tuple(-v / coefficient for v in observed))
                    (lower if coefficient > 0 else upper).add(certificate)
                else:
                    certificate = lift(observed)
                    values = full.T * sp.Matrix(certificate)
                    if any(value < 0 for value in values):
                        raise RuntimeError("Invalid exact data-compatibility certificate.")
                    expression = sp.expand(sum(v * b for v, b in zip(certificate, rhs)))
                    if expression.free_symbols:
                        compatibility.add(sp.Ge(expression, 0))

        # Independently verify every lifted bound over every response type.
        target = sp.Matrix(self.target)
        for certificates, sign in ((lower, -1), (upper, 1)):
            for certificate in certificates:
                residual = full.T * sp.Matrix(certificate) - target
                if any(sign * value < 0 for value in residual):
                    raise RuntimeError("Invalid exact symbolic bound certificate.")
        return self._make_bounds(
            lower, upper, equalities,
            compatibility=tuple(sorted(compatibility, key=sp.default_sort_key)))

    def _make_bounds(self, lower, upper, equalities, *, compatibility=()):
        lower, upper = tuple(sorted(lower)), tuple(sorted(upper))
        if not lower or not upper:
            raise RuntimeError("Exact symbolic enumeration failed to find both envelopes.")
        rhs = (sp.Integer(1),) + self.symbols

        def expressions(certificates):
            return tuple(sp.expand(sum(v * b for v, b in zip(cert, rhs)))
                         for cert in certificates)

        return SymbolicBounds(
            expressions(lower), expressions(upper), self.symbols,
            lower, upper, equalities,
            compatibility_inequalities=compatibility)

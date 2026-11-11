"""Exact hull backend for unrestricted linear response-distribution models."""
from dataclasses import dataclass
from fractions import Fraction
import sympy as sp
from ..model import Polynomial, Term


class CddBackend:
    """Accept only normalized response mixtures with symbolic affine data rows."""
    @staticmethod
    def supports(model):
        try:
            _response_view(model)
        except ValueError:
            return False
        return True

    def solve(self, model):
        return _response_view(model).solve()


def _response_view(model):
    decisions = tuple(variable for variable in model.variables if variable.kind == 'decision')
    parameters = model.parameters
    if (not decisions or not model.is_linear or model.objective.constant
            or any(variable.lower != 0 or variable.upper != 1 for variable in decisions)
            or any(variable.lower is not None or variable.upper is not None for variable in parameters)
            or any(row.relation != '==' for row in model.constraints)
            or len(model.constraints) != len(parameters) + 1):
        raise ValueError('Exact cdd requires unrestricted normalized linear response mixtures')
    names = tuple(variable.name for variable in decisions)
    normalization = Polynomial(tuple(Term(1, (name,)) for name in names) + (Term(-1),))
    rows = [row for row in model.constraints if row.expression != normalization]
    if len(rows) != len(parameters):
        raise ValueError('Exact cdd requires one normalization equality')
    matrix = []
    for parameter in parameters:
        matching = [row for row in rows if any(term.variables == (parameter.name,) and term.coefficient == -1
                                              for term in row.expression.terms)]
        if len(matching) != 1:
            raise ValueError('Exact cdd requires one defining row per symbolic parameter')
        row = matching[0]
        coefficients = {term.variables: term.coefficient for term in row.expression.terms}
        if any(variables not in {(name,) for name in names + (parameter.name,)}
               for variables in coefficients):
            raise ValueError('Exact cdd data rows must contain only response weights and their parameter')
        matrix.append(tuple(sp.Rational(coefficients.get((name,), 0)) for name in names))
    objective = {term.variables: term.coefficient for term in model.objective.terms}
    if any(variables not in {(name,) for name in names} for variables in objective):
        raise ValueError('Exact cdd objective must involve only response weights')
    target = tuple(sp.Rational(objective.get((name,), 0)) for name in names)
    symbols = tuple(sp.Symbol(variable.label or variable.name, real=True) for variable in parameters)
    if len(set(symbols)) != len(symbols):
        raise ValueError('Symbolic data labels must be unique')
    return _ResponseView(tuple(matrix), target, symbols)


@dataclass(frozen=True)
class _ResponseView:
    matrix: tuple
    target: tuple
    symbols: tuple

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

        from ..symbolic_results import SymbolicBounds
        return SymbolicBounds(
            expressions(lower), expressions(upper), self.symbols,
            lower, upper, equalities,
            compatibility_inequalities=compatibility)

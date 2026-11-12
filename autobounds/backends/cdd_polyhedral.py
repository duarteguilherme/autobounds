"""Exact projection of rational affine LPs to (data parameters, objective).

Unlike the fast response-mixture backend, this supports arbitrary affine
equalities, inequalities, and bounds. It enumerates the full polyhedron's
generators before projecting, so it can be expensive. Parameter-dependent
decision coefficients are not affine in the joint coordinates and are rejected.
"""
from fractions import Fraction

import sympy as sp

from ..model import Polynomial, Term
from ..exceptions import UnsupportedModelError
from ..symbolic_results import SymbolicBounds


class CddPolyhedralBackend:
    @staticmethod
    def supports(model):
        return model.is_linear

    def solve(self, model):
        if not self.supports(model):
            raise UnsupportedModelError('Exact polyhedral cdd requires a jointly affine model; '
                             'nonlinear terms and parameter-dependent decision '
                             'coefficients need further reduction or parameter binding.')
        try:
            import cdd.gmp as cdd
        except ImportError as exc:
            raise ImportError("Install exact cdd with: python -m pip install 'autobounds[symbolic]'") from exc
        names = tuple(v.name for v in model.variables)
        position = {name: i+1 for i, name in enumerate(names)}
        target_index = len(names)+1
        rows, equalities = [], set()

        def append(expression, relation):
            row = [Fraction(0)] * (len(names)+2)
            for term in expression.terms:
                index = position[term.variables[0]] if term.variables else 0
                row[index] += term.coefficient
            if relation == '<=':
                row = [-value for value in row]
            if relation == '==':
                equalities.add(len(rows))
            rows.append(row)

        for constraint in model.constraints:
            append(constraint.expression, constraint.relation)
        for v in model.variables:
            if v.lower is not None:
                append(Polynomial((Term(1, (v.name,)), Term(-v.lower))), '>=')
            if v.upper is not None:
                append(Polynomial((Term(-1, (v.name,)), Term(v.upper))), '>=')
        append(model.objective, '==')
        rows[-1][target_index] = -1
        matrix = cdd.matrix_from_array(rows, lin_set=equalities, rep_type=cdd.RepType.INEQUALITY)
        generators = cdd.copy_generators(cdd.polyhedron_from_matrix(matrix))
        if not generators.array:
            raise ValueError('Infeasible linear model: cdd found no feasible generators')
        projected, lines = [], set()
        indices = [0] + [position[v.name] for v in model.parameters] + [target_index]
        for i, row in enumerate(generators.array):
            point = [row[index] for index in indices]
            if not any(point):
                continue  # a ray entirely in eliminated directions
            if i in generators.lin_set:
                lines.add(len(projected))
            projected.append(point)
        if not projected:
            raise RuntimeError('Projection has no generators')
        projected = cdd.matrix_from_array(projected, lin_set=lines, rep_type=cdd.RepType.GENERATOR)
        hull = cdd.copy_inequalities(cdd.polyhedron_from_matrix(projected))
        symbols = tuple(sp.Symbol(v.label or v.name, real=True) for v in model.parameters)
        if len(set(symbols)) != len(symbols):
            raise ValueError('Symbolic parameter labels must be unique')
        rhs = (sp.Integer(1),) + symbols
        lower, upper, compatible_eq, compatible_ge = set(), set(), set(), set()
        for i, row in enumerate(hull.array):
            row = tuple(sp.Rational(v.numerator, v.denominator) for v in row)
            coefficient = row[-1]
            if coefficient:
                certificate = tuple(-v/coefficient for v in row[:-1])
                if i in hull.lin_set:
                    lower.add(certificate)
                    upper.add(certificate)
                else:
                    (lower if coefficient > 0 else upper).add(certificate)
            else:
                expression = sp.expand(sum(a*b for a, b in zip(row[:-1], rhs)))
                condition = sp.Eq(expression, 0) if i in hull.lin_set else sp.Ge(expression, 0)
                if condition != sp.true:
                    (compatible_eq if i in hull.lin_set else compatible_ge).add(condition)
        lower, upper = tuple(sorted(lower)), tuple(sorted(upper))

        def expressions(certificates, unbounded):
            return tuple(sp.expand(sum(a*b for a, b in zip(c, rhs)))
                         for c in certificates) or (unbounded,)

        return SymbolicBounds(
            expressions(lower, -sp.oo), expressions(upper, sp.oo), symbols,
            lower, upper, tuple(sorted(compatible_eq, key=sp.default_sort_key)),
            method='exact_cdd_projection',
            compatibility_inequalities=tuple(sorted(compatible_ge, key=sp.default_sort_key)))

"""Exact Farkas projection with rational coefficients and symbolic RHS."""
from fractions import Fraction

import sympy as sp


def project_inequalities(matrix, rhs):
    """Eliminate unrestricted x from A x <= rhs using exact dual cone rays.

    A must be a nonempty rectangular rational matrix. RHS expressions may
    depend nonlinearly on retained symbols. Return (conditions, certificates),
    where each condition is lambda.T * rhs >= 0, lambda >= 0 and A.T*lambda=0.
    This avoids enumerating the primal polyhedron for every parameter value.
    """
    import cdd.gmp as cdd

    rows = [tuple(Fraction(v) for v in row) for row in matrix]
    m = len(rows)
    if not m or len(rhs) != m or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('Expected a nonempty rectangular matrix and matching RHS')
    cone = [[0] + [int(i == j) for j in range(m)] for i in range(m)]
    equalities = set()
    for j in range(len(rows[0])):
        equalities.add(len(cone))
        cone.append([0] + [rows[i][j] for i in range(m)])
    h = cdd.matrix_from_array(cone, lin_set=equalities,
                              rep_type=cdd.RepType.INEQUALITY)
    generators = cdd.copy_generators(cdd.polyhedron_from_matrix(h))
    certificates = []
    for ray in generators.array:
        if ray[0] or not any(ray[1:]):
            continue
        values = tuple(sp.Rational(v.numerator, v.denominator) for v in ray[1:])
        certificates.append(values)
    certificates = tuple(sorted(certificates))
    conditions = tuple(sp.expand(sum(v*r for v, r in zip(ray, rhs)))
                       for ray in certificates)
    return conditions, certificates

"""Independent, manually reduced oracle for regression tests only.

Production SymbolicProblem must never import or call this case-specific model.
"""
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from autobounds.backends.cdd_projection import project_inequalities


@dataclass(frozen=True)
class HomogeneousIVBounds:
    """Natural-Z averaged ATE, homogeneous across nonempty D principal strata.

    Inputs: a=P(D=1|Z=0), n=P(D=0|Z=1), c=1-a-n, w=P(Z=1),
    g0=P(D=1,Y=1|Z=0), g1=P(D=0,Y=1|Z=1),
    b0=P(D=0,Y=1|Z=0), b1=P(D=1,Y=1|Z=1).
    ``inequalities`` are exact expressions required to be nonnegative;
    certificates multiply the original RHS rows and annihilate their matrix.
    Zero-mass strata impose no conditional-effect restriction.
    """
    symbols: tuple
    target: object
    matrix: tuple
    rhs: tuple
    inequalities: tuple
    certificates: tuple
    lower_expressions: tuple
    upper_expressions: tuple
    method: str = 'exact_cdd_farkas_projection'

    @property
    def lower(self):
        return sp.Max(*self.lower_expressions)

    @property
    def upper(self):
        return sp.Min(*self.upper_expressions)

    def evaluate(self, probabilities):
        """Validate data and evaluate bounds, including absent principal strata."""
        values = {str(k): sp.Rational(str(v)) for k, v in probabilities.items()}
        if 'c' not in values and 'a' in values and 'n' in values:
            values['c'] = 1-values['a']-values['n']
        missing = {str(s) for s in self.symbols} - values.keys()
        if missing:
            raise ValueError(f'Missing probabilities: {sorted(missing)}')
        a, n, c, w, g0, g1, b0, b1 = [values[str(s)] for s in self.symbols]
        if (min(a, n, c) < 0 or a+n+c != 1 or not 0 < w < 1
                or not 0 <= g0 <= a or not 0 <= g1 <= n
                or not 0 <= b0 <= n+c or not 0 <= b1 <= a+c):
            raise ValueError('Invalid probabilities or violation of no defiers')
        substitutions = {s: values[str(s)] for s in self.symbols}
        lower, upper = sp.Integer(-1), sp.Integer(1)
        for expression in self.inequalities:
            expression = expression.subs(substitutions)
            coefficient = expression.coeff(self.target)
            constant = expression.subs(self.target, 0)
            if coefficient > 0:
                lower = max(lower, -constant/coefficient)
            elif coefficient < 0:
                upper = min(upper, -constant/coefficient)
            elif constant < 0:
                raise ValueError('Data incompatible with homogeneous effects')
        if lower > upper:
            raise ValueError('Data incompatible with homogeneous effects')
        return float(lower), float(upper)


@lru_cache(maxsize=1)
def homogeneous_iv_bounds():
    """Symbolically solve the binary no-defiers, homogeneous-effect IV model.

    Z is independent of latent response types; Z->Y is allowed. Homogeneity
    concerns E[Y(1,Z)-Y(0,Z)|D(0),D(1)], averaging over natural Z. There is
    no equality assumption on the effects at each separate value of Z.

    Only two unknown observed-outcome allocations need elimination:
    U=(1-w)P(N,Y(0,0)=1), V=w P(A,Y(1,1)=1). All other potential-outcome
    marginals can be freely coupled within each principal stratum.
    """
    a, n, c, w = sp.symbols('a n c w', nonnegative=True)
    g0, g1, b0, b1 = sp.symbols('g0 g1 b0 b1', nonnegative=True)
    t = sp.Symbol('ATE', real=True)
    # Each row is A*(U,V) <= rhs. Coefficients of eliminated variables
    # are constant even though the retained ATE has data-dependent coefficients.
    matrix = ((-1, 0), (-1, 0), (1, 0), (1, 0),
              (-1, 0), (1, 0),
              (0, -1), (0, -1), (0, 1), (0, 1),
              (0, -1), (0, 1), (-1, 1), (1, -1))
    rhs = (0, -(1-w)*(b0-c), (1-w)*n, (1-w)*b0,
           w*g1+n*t, n-w*g1-n*t,
           0, -w*(b1-c), w*a, w*b1,
           (1-w)*g0-a*t, a*t+a-(1-w)*g0,
           -c*t-(1-w)*b0+w*b1+(1-w)*c,
           c*t+(1-w)*b0-w*b1+w*c)
    inequalities, certificates = project_inequalities(matrix, rhs)
    lower, upper = {sp.Integer(-1)}, {sp.Integer(1)}
    for expression in inequalities:
        coefficient = expression.coeff(t)
        if coefficient == 0:
            continue
        constant = expression.subs(t, 0)
        # Cone rays give sums of nonnegative stratum masses with one sign.
        if coefficient.is_nonnegative:
            lower.add(sp.Piecewise((-constant/coefficient, coefficient > 0),
                                   (-sp.oo, True)))
        elif coefficient.is_nonpositive:
            upper.add(sp.Piecewise((-constant/coefficient, coefficient < 0),
                                   (sp.oo, True)))
        else:
            raise RuntimeError('Unexpected mixed-sign target coefficient')
    return HomogeneousIVBounds((a, n, c, w, g0, g1, b0, b1), t,
                               matrix, tuple(map(sp.sympify, rhs)), inequalities,
                               certificates, tuple(sorted(lower, key=sp.default_sort_key)),
                               tuple(sorted(upper, key=sp.default_sort_key)))

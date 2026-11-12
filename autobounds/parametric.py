"""Automatic response-coordinate reduction and exact parametric LP elimination.

A rational convex hull discovers the feasible region of the linear response
features actually used by the declarations. Symbolic elimination then retains
parameters and the target, with divisions allowed only by proved nonzero terms.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

import sympy as sp

from .exceptions import UnsupportedModelError
from .model import Constraint, MathematicalModel, Polynomial, Term, Variable


def _fresh(prefix, occupied):
    while prefix in occupied:
        prefix += '_'
    occupied.add(prefix)
    return sp.Symbol(prefix, real=True)


def _polynomial(expression):
    expression = sp.Poly(sp.expand(expression), *sorted(expression.free_symbols, key=str)) if expression.free_symbols else None
    if expression is None:
        raise ValueError('Use _model for constant expressions')
    return Polynomial(tuple(Term(Fraction(coef), tuple(str(v) for v, power in zip(expression.gens, powers)
                                                       for _ in range(power)))
                            for powers, coef in expression.terms()))


def _model(decisions, parameters, equalities, inequalities, target):
    def polynomial(e):
        e = sp.expand(e)
        return _polynomial(e) if e.free_symbols else Polynomial((Term(Fraction(e)),))
    return MathematicalModel(
        tuple(Variable(str(v), None, None) for v in decisions)
        + tuple(Variable(str(p), None, None, 'parameter', str(p)) for p in parameters),
        tuple(Constraint(polynomial(e), '==') for e in equalities)
        + tuple(Constraint(polynomial(e), '>=') for e in inequalities),
        polynomial(target))


class _Signs:
    """Exact factor signs from explicitly positive expressions; never sample."""
    def __init__(self, positive):
        self.positive = tuple(sp.expand(e) for e in positive)

    def sign(self, expression):
        expression = sp.factor(expression)
        if not expression.free_symbols:
            return int(sp.sign(expression))
        numerator, denominator = sp.fraction(expression)
        result = 1
        for part in (numerator, denominator):
            coefficient, factors = sp.factor_list(part)
            result *= int(sp.sign(coefficient))
            for factor, power in factors:
                found = None
                for known in self.positive:
                    ratio = sp.cancel(factor/known)
                    if ratio.is_number and ratio != 0:
                        found = int(sp.sign(ratio))
                        break
                if found is None:
                    return None
                result *= found**power
        return result

    def normalize(self, expression):
        """Remove only strictly signed common factors, preserving >= 0."""
        numerator, denominator = sp.fraction(sp.cancel(expression))
        denominator_sign = self.sign(denominator)
        if denominator_sign not in (-1, 1):
            raise UnsupportedModelError('Cannot certify the sign of a symbolic denominator')
        coefficient, factors = sp.factor_list(numerator)
        result = sp.sign(coefficient)*denominator_sign
        for factor, power in factors:
            sign = self.sign(factor)
            result *= sign**power if sign in (-1, 1) else factor**power
        return sp.expand(result)


def _unique(expressions):
    return list(dict.fromkeys(e for e in expressions if e != 0))


def _affine_eliminate(variables, equalities, inequalities, signs):
    variables, eq, ge = list(variables), list(equalities), list(inequalities)
    steps = []
    while True:
        candidates = []
        for i, e in enumerate(eq):
            e = sp.expand(e)
            for v in variables:
                coefficient = e.coeff(v)
                if (coefficient and not coefficient.free_symbols.intersection(variables)
                        and signs.sign(coefficient) in (-1, 1)):
                    candidates.append((bool(coefficient.free_symbols), sp.count_ops(e), str(v), i, v, coefficient))
        if not candidates:
            break
        _, _, _, i, v, coefficient = min(candidates)
        replacement = sp.cancel(v-eq.pop(i)/coefficient)
        eq = _unique([signs.normalize(e.subs(v, replacement)) for e in eq])
        ge = _unique([signs.normalize(e.subs(v, replacement)) for e in ge])
        variables.remove(v)
        steps.append((str(v), replacement))
    return variables, eq, ge, tuple(steps)


@dataclass(frozen=True)
class BoundCandidate:
    """One endpoint expression and the parameter region where it applies."""
    expression: object
    active_when: object


@dataclass(frozen=True)
class ParametricBounds:
    """Exact projected conditions and sign-conditional symbolic envelopes."""
    symbols: tuple
    target: object
    equalities: tuple
    inequalities: tuple
    positive: tuple
    report: dict
    method: str = 'automatic_response_hull_and_symbolic_projection'

    @cached_property
    def _nonnegative_expressions(self):
        # Target-free projected rows certify signs on the compatibility region.
        # Positive rational rescaling does not change a sign certificate.
        return {sp.expand(e).as_content_primitive()[1]
                for e in self.inequalities if not e.has(self.target)}

    def _coefficient_sign(self, coefficient):
        if coefficient.is_positive:
            return 1
        if coefficient.is_negative:
            return -1
        primitive = sp.expand(coefficient).as_content_primitive()[1]
        if primitive in self._nonnegative_expressions:
            return 1
        if -primitive in self._nonnegative_expressions:
            return -1
        return None

    def _candidates(self, lower):
        candidates = []
        for equality, rows in ((False, self.inequalities), (True, self.equalities)):
            for e in rows:
                coefficient = e.coeff(self.target)
                if coefficient == 0:
                    continue
                sign = self._coefficient_sign(coefficient)
                if not equality and sign is not None and (sign > 0) != lower:
                    continue
                condition = (sp.Ne(coefficient, 0) if equality else
                             coefficient > 0 if lower else coefficient < 0)
                expression = sp.factor(-e.subs(self.target, 0)/coefficient)
                candidates.append(BoundCandidate(expression, condition))
        return tuple(sorted(dict.fromkeys(candidates),
                            key=lambda c: (sp.count_ops(c.expression),
                                           sp.default_sort_key(c.expression),
                                           sp.default_sort_key(c.active_when))))

    @cached_property
    def lower_candidates(self):
        """Numberable lower candidates, with zero-denominator guards."""
        return self._candidates(True)

    @cached_property
    def upper_candidates(self):
        """Numberable upper candidates, with zero-denominator guards."""
        return self._candidates(False)

    @cached_property
    def compatibility_conditions(self):
        """Data restrictions, including restrictions when a coefficient vanishes.

        These conditions and lower <= upper define the compatible parameter
        region; candidate sign certificates are valid on that region.
        """
        conditions = [e > 0 for e in self.positive]
        for equality, rows in ((False, self.inequalities), (True, self.equalities)):
            for e in rows:
                coefficient = e.coeff(self.target)
                constant = e.subs(self.target, 0)
                condition = sp.Eq(constant, 0) if equality else constant >= 0
                if coefficient != 0:
                    condition = sp.Or(sp.Ne(coefficient, 0), condition)
                if condition != sp.true:
                    conditions.append(condition)
        return tuple(dict.fromkeys(conditions))

    def format_candidates(self, *, aliases=None, style='text', include_compatibility=False):
        """Show numbered endpoint candidates, separately from compatibility.

        ``style='latex'`` uses LaTeX display equations in the text report. Aliases map
        probability labels to distinct short names; unspecified names default
        to p0, p1, ... . Renaming affects presentation only.
        """
        if style not in ('text', 'latex'):
            raise ValueError("style must be 'text' or 'latex'")
        aliases = {} if aliases is None else {str(k): str(v) for k, v in aliases.items()}
        if set(aliases) - {str(p) for p in self.symbols}:
            raise ValueError('Aliases contain unknown probability labels')
        names = [aliases.get(str(p), f'p{i}') for i, p in enumerate(self.symbols)]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError('Probability aliases must be nonempty and distinct')
        substitutions = {p: sp.Symbol(name, real=True) for p, name in zip(self.symbols, names)}
        render = sp.latex if style == 'latex' else str
        lines = ['Probability labels:']
        lines += [f'  {name} = {p}' for p, name in zip(self.symbols, names)]
        for prefix, title, candidates in [('L', 'Lower', self.lower_candidates),
                                          ('U', 'Upper', self.upper_candidates)]:
            lines += ['', f'{title} candidates ({len(candidates)}):']
            for i, candidate in enumerate(candidates, 1):
                expression = render(candidate.expression.xreplace(substitutions))
                condition = candidate.active_when.xreplace(substitutions)
                if style == 'latex':
                    suffix = '' if condition == sp.true else r'\quad\text{if }' + render(condition)
                    lines.append(r'\[' + f'{prefix}_{{{i}}} = ' + expression + suffix + r'\]')
                else:
                    suffix = '' if condition == sp.true else f'  [if {condition}]'
                    lines.append(f'  {prefix}{i} = {expression}{suffix}')
            if candidates:
                operation = 'max' if prefix == 'L' else 'min'
                lines.append(f'{title} bound = {operation} of the applicable {prefix} candidates.')
            else:
                lines.append(f'{title} bound = ' + ('-infinity' if prefix == 'L' else '+infinity'))
        lines += ['', 'A candidate with a zero denominator is inactive.',
                  'Compatibility conditions must hold, including zero-coefficient cases,',
                  'and the lower bound must not exceed the upper bound.']
        if include_compatibility:
            lines += ['', 'Compatibility conditions:']
            lines += [f'  C{i} = {render(e.xreplace(substitutions))}'
                      for i, e in enumerate(self.compatibility_conditions, 1)]
        else:
            lines.append('Use compatibility_conditions or include_compatibility=True to inspect them.')
        return '\n'.join(lines)

    def _expressions(self, lower):
        output = []
        for e in self.inequalities:
            coefficient = e.coeff(self.target)
            if coefficient == 0:
                continue
            condition = coefficient > 0 if lower else coefficient < 0
            output.append(sp.Piecewise((-e.subs(self.target, 0)/coefficient, condition),
                                       (-sp.oo if lower else sp.oo, True)))
        for e in self.equalities:
            coefficient = e.coeff(self.target)
            if coefficient != 0:
                output.append(sp.Piecewise((-e.subs(self.target, 0)/coefficient, sp.Ne(coefficient, 0)),
                                           (-sp.oo if lower else sp.oo, True)))
        return tuple(output) or ((-sp.oo if lower else sp.oo),)

    @property
    def lower_expressions(self):
        return self._expressions(True)

    @property
    def upper_expressions(self):
        return self._expressions(False)

    @property
    def lower(self):
        return sp.Max(*self.lower_expressions, evaluate=False)

    @property
    def upper(self):
        return sp.Min(*self.upper_expressions, evaluate=False)

    def evaluate(self, probabilities):
        values = {str(k): sp.Rational(str(v)) for k, v in probabilities.items()}
        missing = {str(s) for s in self.symbols} - values.keys()
        if missing:
            raise ValueError(f'Missing probabilities: {sorted(missing)}')
        sub = {p: values[str(p)] for p in self.symbols}
        if any(e.subs(sub) <= 0 for e in self.positive):
            raise ValueError('Conditional data require strictly positive root probabilities')
        lower, upper = -sp.oo, sp.oo
        for e, equality in [(e, False) for e in self.inequalities] + [(e, True) for e in self.equalities]:
            e = sp.expand(e.subs(sub))
            coefficient, constant = e.coeff(self.target), e.subs(self.target, 0)
            if coefficient == 0:
                if (constant != 0 if equality else constant < 0):
                    raise ValueError('Data incompatible with the declared model')
            else:
                value = -constant/coefficient
                if equality or coefficient > 0:
                    lower = max(lower, value)
                if equality or coefficient < 0:
                    upper = min(upper, value)
        if lower > upper:
            raise ValueError('Data incompatible with the declared model')
        return float(lower), float(upper)


@dataclass(frozen=True)
class ParametricReduction:
    model: MathematicalModel
    variables: tuple
    parameters: tuple
    target: object
    equalities: tuple
    inequalities: tuple
    positive: tuple
    report: dict
    feature_map: tuple = ()
    hull_rows: tuple = ()

    def solve(self, *, max_rows=10000):
        """Fourier–Motzkin with certified signs and explicit growth limits."""
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
            raise ValueError('max_rows must be a positive integer')
        signs = _Signs(self.positive)
        variables = list(self.variables)
        eq = list(self.equalities)
        ge = list(self.inequalities)
        retained_eq = []
        for e in eq:
            if e.free_symbols.intersection(variables):
                ge.extend([e, -e])
            else:
                retained_eq.append(e)
        ge = _unique([signs.normalize(e) for e in ge])
        stages = []
        while variables:
            choices = []
            for v in variables:
                pos, neg, zero = [], [], []
                for e in ge:
                    coefficient = e.coeff(v)
                    sign = signs.sign(coefficient)
                    if sign is None:
                        break
                    if sign == 0:
                        zero.append(e)
                    else:
                        (pos if sign > 0 else neg).append((coefficient, e-coefficient*v))
                else:
                    choices.append((len(pos)*len(neg), str(v), v, pos, neg, zero))
            if not choices:
                raise UnsupportedModelError(
                    'Cannot certify nuisance-coefficient signs; parameter case splitting is required. '
                    'This symbolic backend cannot return bounds without resolving those signs.')
            count, _, v, pos, neg, zero = min(choices, key=lambda item: item[:2])
            if count+len(zero) > max_rows:
                raise ValueError(
                    f'Symbolic projection exceeds max_rows={max_rows}: '
                    f'the next step would generate {count+len(zero)} rows. '
                    'Increase max_rows to allow this step; no relaxation was returned.')
            ge = _unique([signs.normalize(a*d-c*b) for a, b in pos for c, d in neg] + zero)
            variables.remove(v)
            stages.append((str(v), len(pos), len(neg), len(ge)))
        report = dict(self.report, projection_steps=tuple(stages), projected_inequalities=len(ge))
        return ParametricBounds(self.parameters, self.target, tuple(retained_eq), tuple(ge), self.positive, report)


class ParametricResponseProgram:
    """Graph-independent polynomial declarations over independent simplices."""
    def __init__(self, components, data, constraints, zeros, estimand, homogeneity, positive_roots):
        self.components = components
        self.data = data
        self.constraints = constraints
        self.zeros = zeros
        self.estimand = estimand
        self.homogeneity = homogeneity
        self.positive_roots = positive_roots

    def linearize(self):
        import cdd.gmp as cdd

        occupied = {str(q) for c in self.components for q in c} | set(self.data) | set(self.constraints)
        target = _fresh('_ab_target', occupied)
        parameters = tuple(sp.Symbol(label, real=True) for label in (*self.data, *self.constraints))
        labeled = list(zip(parameters, list(self.data.values())+list(self.constraints.values())))
        removed = set()
        for e in self.zeros:
            e = sp.expand(e)
            if e.subs({q: 0 for c in self.components for q in c}) != 0:
                raise ValueError('Zero events must be nonnegative response indicators')
            for term in sp.Add.make_args(e):
                coefficient, rest = term.as_coeff_Mul()
                if coefficient <= 0 or not isinstance(rest, sp.Symbol):
                    raise ValueError('Zero events must be nonnegative linear response indicators')
                removed.add(rest)
        substitutions = {q: 0 for q in removed}
        components = [tuple(q for q in c if q not in removed) for c in self.components]
        if any(not c for c in components):
            raise ValueError('Zero restrictions exclude every response type in a component')
        # Bind fully observed independent components by exact linear algebra.
        unbound = []
        for component in components:
            rows = [sum(component)-1]
            rows += [e.subs(substitutions)-p for p, e in labeled
                     if e.free_symbols and e.free_symbols <= set(component)]
            matrix, rhs = sp.linear_eq_to_matrix(rows, component)
            if matrix.free_symbols or matrix.rank() < len(component):
                unbound.append(component)
                continue
            indices = list(matrix.T.rref()[1])
            solution = matrix[indices, :].inv()*rhs[indices, :]
            substitutions.update(zip(component, solution))
        if not unbound:
            raise UnsupportedModelError(
                'All response components are fully identified. This reduction backend '
                'currently requires one unresolved response component; it cannot '
                'compile this fully identified model.')
        if len(unbound) > 1:
            raise UnsupportedModelError(
                f'Reduction requires one unresolved response component; found {len(unbound)}. '
                'The independent components are not pooled. Supply any missing '
                'root distributions; this backend cannot eliminate multiple unresolved components.')
        q = unbound[0]
        positive = tuple(sp.expand(e.subs(substitutions)) for e in self.positive_roots)
        if any(e.free_symbols.intersection(q) for e in positive):
            raise UnsupportedModelError('Load the root distribution used by conditional data')
        equalities = [sp.expand((e-p).subs(substitutions)) for p, e in labeled]
        equalities += [sp.expand((sum(c)-1).subs(substitutions)) for c in components]
        equalities.append(sp.expand((self.estimand-target).subs(substitutions)))
        for effect, masses, deltas in self.homogeneity:
            # Require a genuine partition, checked on every allowed response.
            ms = [sp.expand(m.subs(substitutions)) for m in masses]
            for response in q:
                values = [m.subs({x: int(x == response) for x in q}) for m in ms]
                if any(v not in (0, 1) for v in values) or sum(values) != 1:
                    raise ValueError('Homogeneity strata must partition the allowed response types')
            average = effect.subs(substitutions)
            # The common effect is the estimand only if it is the declared effect.
            if sp.expand(average-self.estimand.subs(substitutions)) != 0:
                raise UnsupportedModelError('Homogeneity effect must match the declared estimand')
            equalities += [sp.expand(d.subs(substitutions)-m*target) for m, d in zip(ms, deltas)]
        # Split parameter/target coefficients from rational linear q features.
        features = [tuple([sp.Integer(1)]*len(q))]
        for e in equalities:
            poly = sp.Poly(e, *q)
            if poly.total_degree() > 1:
                raise UnsupportedModelError('Unresolved nonlinear response products; cannot certify an LP reduction')
            coeffs = [sp.expand(e).coeff(v) for v in q]
            monomials = set()
            retained = parameters+(target,)
            for coefficient in coeffs:
                monomials.update(sp.Poly(coefficient, *retained).monoms())
            for powers in sorted(monomials):
                row = tuple(sp.Poly(coefficient, *retained).coeff_monomial(powers) for coefficient in coeffs)
                if any(row) and row not in features:
                    features.append(row)
        matrix = sp.Matrix(features)
        matrix = matrix[list(matrix.T.rref()[1]), :]
        pivots = list(matrix.rref()[1])
        inverse = matrix[:, pivots].inv()
        coordinates = tuple(_fresh(f'_ab_feature{i}', occupied) for i in range(matrix.rows))
        transformed = []
        for e in equalities:
            coeffs = sp.Matrix([[e.coeff(v) for v in q]])
            weights = coeffs[:, pivots]*inverse
            if any(sp.expand(v) != 0 for v in coeffs-weights*matrix):
                raise RuntimeError('Response feature projection lost a declared linear form')
            constant = e.subs({v: 0 for v in q})
            transformed.append(sp.expand(constant+sum(w*x for w, x in zip(weights, coordinates))))
        points = [[1]+[Fraction(v) for v in matrix[:, j]] for j in range(len(q))]
        hull = cdd.copy_inequalities(cdd.polyhedron_from_matrix(cdd.matrix_from_array(points, rep_type=cdd.RepType.GENERATOR)))
        inequalities = []
        hull_rows = []
        for i, row in enumerate(hull.array):
            e = sp.Rational(row[0])+sum(sp.Rational(a)*v for a, v in zip(row[1:], coordinates))
            (transformed if i in hull.lin_set else inequalities).append(e)
            hull_rows.append((e, '==' if i in hull.lin_set else '>='))
        signs = _Signs(positive)
        variables, eq, ge, steps = _affine_eliminate(coordinates, transformed, inequalities, signs)
        # LP means linear in decisions for fixed data, not jointly affine.
        if any(sp.Poly(e, *(tuple(variables)+(target,))).total_degree() > 1 for e in eq+ge):
            raise UnsupportedModelError('Reduced problem is still nonlinear in decision variables')
        report = dict(response_types=len(q), removed_response_types=len(removed),
                      response_features=matrix.rows, hull_constraints=len(hull.array),
                      affine_substitutions=steps, nuisance_variables=len(variables),
                      is_parametric_linear=True)
        model = _model(tuple(variables)+(target,), parameters, eq, ge, target)
        feature_map = tuple((x, sum(a*v for a, v in zip(matrix.row(i), q)))
                            for i, x in enumerate(coordinates))
        return ParametricReduction(model, tuple(variables), parameters, target,
                                   tuple(eq), tuple(ge), positive, report,
                                   feature_map, tuple(hull_rows))

    def compile(self):
        """Return the automatically reduced mathematical model."""
        return self.linearize().model

    def solve(self, **options):
        return self.linearize().solve(**options)

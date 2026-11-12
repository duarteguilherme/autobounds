"""Exact, bounded-cost reductions shared by numerical and symbolic solvers.

Every accepted rewrite preserves the feasible set under the recorded change
of variables. Failure to prove a precondition leaves the relevant expression
unchanged. No product is replaced by an independent unconstrained variable.
"""
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction

from .model import Constraint, MathematicalModel, Polynomial, Term, Variable
from .support import eliminate_zero_variables


def constant(value):
    return Polynomial((Term(value),))


def variable(name):
    return Polynomial((Term(1, (name,)),))


def add(*expressions):
    return Polynomial(tuple(t for p in expressions for t in p.terms))


def scale(expression, coefficient):
    return Polynomial(tuple(Term(t.coefficient * coefficient, t.variables)
                            for t in expression.terms))


class ExpansionLimit(Exception):
    """A sufficient reduction exceeded its resource budget."""


def multiply(left, right, budget):
    if len(left.terms) * len(right.terms) > budget:
        raise ExpansionLimit
    result = Polynomial(tuple(Term(a.coefficient*b.coefficient,
                                   a.variables+b.variables)
                              for a in left.terms for b in right.terms))
    if len(result.terms) > budget:
        raise ExpansionLimit
    return result


def substitute(expression, name, replacement, budget):
    terms = []
    for term in expression.terms:
        power = term.variables.count(name)
        piece = Polynomial((Term(term.coefficient,
                                 tuple(v for v in term.variables if v != name)),))
        for _ in range(power):
            piece = multiply(piece, replacement, budget)
        terms.extend(piece.terms)
        if len(terms) > budget:
            raise ExpansionLimit
    return Polynomial(tuple(terms))


def expression_bounds(expression, model):
    """Conservative exact interval, strengthened by proportional scalar rows."""
    domains = {v.name: v for v in model.variables}
    lower = upper = Fraction(0)
    for term in expression.terms:
        lo = hi = term.coefficient
        for name, power in Counter(term.variables).items():
            v = domains[name]
            if v.lower is None or v.upper is None:
                lo = hi = None
                break
            ends = [v.lower**power, v.upper**power]
            if power % 2 == 0 and v.lower <= 0 <= v.upper:
                ends.append(Fraction(0))
            products = [a*b for a in (lo, hi) for b in (min(ends), max(ends))]
            lo, hi = min(products), max(products)
        lower = lower+lo if lower is not None and lo is not None else None
        upper = upper+hi if upper is not None and hi is not None else None
    coefficients = {t.variables: t.coefficient for t in expression.terms if t.variables}
    if not coefficients:
        return expression.constant, expression.constant
    key = next(iter(coefficients))
    for row in model.constraints:
        other = {t.variables: t.coefficient for t in row.expression.terms if t.variables}
        if other.keys() != coefficients.keys():
            continue
        ratio = other[key]/coefficients[key]
        if any(other[k] != ratio*v for k, v in coefficients.items()):
            continue
        bound = expression.constant-row.expression.constant/ratio
        relation = row.relation
        if ratio < 0:
            relation = {'<=': '>=', '>=': '<=', '==': '=='}[relation]
        if relation in ('==', '>='):
            lower = bound if lower is None else max(lower, bound)
        if relation in ('==', '<='):
            upper = bound if upper is None else min(upper, bound)
    return lower, upper


@dataclass(frozen=True)
class Reconstruction:
    name: str
    numerator: Polynomial
    denominator: Polynomial = Polynomial((Term(1),))


@dataclass(frozen=True)
class ReductionResult:
    original: MathematicalModel
    model: MathematicalModel
    reconstruction: tuple = ()
    steps: tuple = ()

    @property
    def report(self):
        def counts(model):
            return {'variables': len(model.variables), 'constraints': len(model.constraints),
                    'nonlinear_constraints': sum(c.expression.degree > 1 for c in model.constraints),
                    'terms': sum(len(c.expression.terms) for c in model.constraints),
                    'objective_degree': model.objective.degree}
        return {'before': counts(self.original), 'after': counts(self.model),
                'is_linear': self.model.is_linear, 'steps': self.steps}

    def restore_solution(self, values):
        restored = self.model.restore_solution(values)
        for rule in reversed(self.reconstruction):
            denominator = rule.denominator.evaluate(restored)
            if denominator == 0:
                raise ValueError('Cannot reconstruct a solution with zero scale')
            restored[rule.name] = float(rule.numerator.evaluate(restored)/denominator)
        restored = self.original.restore_solution(restored)
        names = {v.name for v in self.original.variables + self.original.zero_variables}
        return {name: restored[name] for name in names}

    def restore_result(self, result):
        if result.dgps is None:
            return result
        dgps = {}
        for endpoint, entry in result.dgps.items():
            entry = dict(entry)
            if entry.get('status') == 'ok':
                entry['values'] = self.restore_solution(entry['values'])
            dgps[endpoint] = entry
        return replace(result, dgps=dgps)

    def solve(self, backend='scip', **options):
        if backend == 'scip':
            from .backends.scip import ScipBackend
            return self.restore_result(ScipBackend().solve(self.model, **options))
        if backend == 'cdd':
            from .backends.cdd import CddBackend
            from .backends.cdd_polyhedral import CddPolyhedralBackend
            solver = CddBackend() if CddBackend.supports(self.model) else CddPolyhedralBackend()
            return solver.solve(self.model, **options)
        raise ValueError("backend must be 'scip' or 'cdd'")


def _linear_basis(model, budget):
    """Sparse rational RREF; symbolic parameters are never pivot variables."""
    nonlinear = Counter(name for row in model.constraints if row.expression.degree > 1
                        for term in row.expression.terms for name in term.variables)
    order = sorted((v.name for v in model.variables if v.kind == 'decision'),
                   key=lambda name: (-nonlinear[name], name))
    rows = [{(t.variables[0] if t.variables else None): t.coefficient
             for t in row.expression.terms}
            for row in model.constraints
            if row.relation == '==' and row.expression.degree <= 1]
    for v in model.variables:
        if v.kind == 'decision' and v.lower is not None and v.lower == v.upper:
            rows.append({v.name: Fraction(1), None: -v.lower})
    pivots = []
    position = 0
    for name in order:
        candidates = [i for i in range(position, len(rows)) if rows[i].get(name)]
        if not candidates:
            continue
        selected = min(candidates, key=lambda i: len(rows[i]))
        rows[position], rows[selected] = rows[selected], rows[position]
        divisor = rows[position][name]
        rows[position] = {k: v/divisor for k, v in rows[position].items() if v}
        for i, row in enumerate(rows):
            if i == position or not row.get(name):
                continue
            factor = row[name]
            updated = dict(row)
            for key, value in rows[position].items():
                updated[key] = updated.get(key, 0)-factor*value
                if not updated[key]:
                    del updated[key]
            if (len(updated) > budget or any(
                    max(v.numerator.bit_length(), v.denominator.bit_length()) > 4096
                    for v in updated.values())):
                raise ExpansionLimit
            rows[i] = updated
        pivots.append(name)
        position += 1
    # Do not amplify tiny inconsistent input-rounding residuals into a new
    # numerical formulation. Preserve the original declarations and report it.
    if any(row.get(None) and not any(k is not None and v for k, v in row.items())
           for row in rows):
        return None
    return tuple((name, Polynomial(tuple(
        Term(-value, () if key is None else (key,))
        for key, value in rows[i].items() if key != name)))
                 for i, name in enumerate(pivots))


def _eliminate(model, name, replacement, budget):
    rows = [Constraint(substitute(c.expression, name, replacement, budget), c.relation)
            for c in model.constraints]
    objective = substitute(model.objective, name, replacement, budget)
    domain = next(v for v in model.variables if v.name == name)
    if domain.lower is not None:
        rows.append(Constraint(add(replacement, constant(-domain.lower)), '>='))
    if domain.upper is not None:
        rows.append(Constraint(add(replacement, constant(-domain.upper)), '<='))
    return replace(model, variables=tuple(v for v in model.variables if v.name != name),
                   constraints=tuple(dict.fromkeys(rows)), objective=objective).without_tautologies()


def _cost(model):
    expressions = [model.objective] + [c.expression for c in model.constraints]
    return (sum(len(t.variables)-1 for p in expressions for t in p.terms if len(t.variables) > 1),
            sum(len(p.terms) for p in expressions))


def _cancel_factors(model):
    """Cancel only factors proved nonzero; inequality cancellation needs a sign."""
    import sympy as sp
    symbols = {v.name: sp.Symbol(v.name) for v in model.variables}
    reverse = {s: n for n, s in symbols.items()}
    nonzero = {v.name for v in model.variables
               if (v.lower is not None and v.lower > 0)
               or (v.upper is not None and v.upper < 0)}
    for row in model.constraints:
        terms = [t for t in row.expression.terms if t.variables]
        if row.relation == '==' and len(terms) == 1 and row.expression.constant:
            nonzero.update(terms[0].variables)

    def polynomial(expr):
        ordered = sorted(expr.free_symbols, key=str)
        if not ordered:
            return constant(Fraction(expr))
        p = sp.Poly(expr, *ordered)
        return Polynomial(tuple(Term(Fraction(coefficient), tuple(
            reverse[s] for s, exponent in zip(ordered, powers) for _ in range(exponent)))
                                for powers, coefficient in p.terms()))

    rows, steps = [], []
    for row in model.constraints:
        p = row.expression
        names = {n for t in p.terms for n in t.variables}
        if p.degree <= 1 or p.degree > 8 or len(p.terms) > 256 or len(names) > 12:
            rows.append(row)
            continue
        expression = sum(sp.Rational(t.coefficient.numerator, t.coefficient.denominator)
                         * sp.prod(symbols[n] for n in t.variables) for t in p.terms)
        coefficient, factors = sp.factor_list(expression)
        kept = coefficient
        relation = row.relation
        cancelled = []
        for factor, power in factors:
            factor_poly = polynomial(factor**power)
            lo, hi = expression_bounds(factor_poly, model)
            positive = lo is not None and lo > 0
            negative = hi is not None and hi < 0
            known_nonzero = (len(factor_poly.terms) == 1
                             and set(factor_poly.terms[0].variables) <= nonzero)
            if positive or negative or (known_nonzero and row.relation == '=='):
                if negative:
                    relation = {'==': '==', '<=': '>=', '>=': '<='}[relation]
                cancelled.append(str(factor**power))
            else:
                kept *= factor**power
        if cancelled:
            rows.append(Constraint(polynomial(sp.expand(kept)), relation))
            steps.append(('cancel_nonzero_factors', tuple(cancelled), row.relation, relation))
        else:
            rows.append(row)
    return replace(model, constraints=tuple(dict.fromkeys(rows))).without_tautologies(), steps


def _fractional(model):
    """Charnes–Cooper for theta*D(x)=N(x), with rational affine LP data.

    Require a certified positive denominator lower bound and finite upper
    bound. The latter gives t >= 1/D_max > 0 and excludes spurious t=0 points.
    Symbolic RHS coefficients would create parameter*t products, so are not
    silently handled as a fixed-matrix parametric LP.
    """
    if model.parameters or model.objective.degree != 1:
        return None
    objective_terms = [t for t in model.objective.terms if t.variables]
    if len(objective_terms) != 1:
        return None
    theta = objective_terms[0].variables[0]
    matches = [c for c in model.constraints if any(theta in t.variables for t in c.expression.terms)]
    if len(matches) != 1 or matches[0].relation != '==':
        return None
    defining = matches[0]
    if any(c.expression.degree > 1 for c in model.constraints if c is not defining):
        return None
    numerator, denominator = [], []
    for t in defining.expression.terms:
        power = t.variables.count(theta)
        if power > 1:
            return None
        if power:
            denominator.append(Term(-t.coefficient, tuple(v for v in t.variables if v != theta)))
        else:
            numerator.append(t)
    numerator, denominator = Polynomial(tuple(numerator)), Polynomial(tuple(denominator))
    if numerator.degree > 1 or denominator.degree != 1:
        return None
    lo, hi = expression_bounds(denominator, model)
    if hi is not None and hi < 0:
        numerator, denominator = scale(numerator, -1), scale(denominator, -1)
        lo, hi = (-hi, -lo if lo is not None else None)
    if lo is None or hi is None or lo <= 0 or hi < lo:
        return None
    occupied = {v.name for v in model.variables + model.zero_variables}

    def fresh(prefix):
        while prefix in occupied:
            prefix += '_'
        occupied.add(prefix)
        return prefix

    tname = fresh('_ab_scale')
    mapping = {v.name: fresh('_ab_scaled_' + v.name) for v in model.variables if v.name != theta}

    def homogenize(expr):
        return Polynomial(tuple(Term(term.coefficient,
                                     (mapping[term.variables[0]],) if term.variables else (tname,))
                                for term in expr.terms))

    variables = [Variable(tname, 1/hi, 1/lo)]
    rows = [Constraint(homogenize(c.expression), c.relation)
            for c in model.constraints if c is not defining]
    reconstruction = []
    for v in model.variables:
        if v.name == theta:
            continue
        y = variable(mapping[v.name])
        variables.append(Variable(mapping[v.name], None, None))
        if v.lower is not None:
            rows.append(Constraint(add(y, scale(variable(tname), -v.lower)), '>='))
        if v.upper is not None:
            rows.append(Constraint(add(y, scale(variable(tname), -v.upper)), '<='))
        reconstruction.append(Reconstruction(v.name, y, variable(tname)))
    rows.append(Constraint(add(homogenize(denominator), constant(-1))))
    target = homogenize(numerator)
    theta_domain = next(v for v in model.variables if v.name == theta)
    if theta_domain.lower is not None:
        rows.append(Constraint(add(target, constant(-theta_domain.lower)), '>='))
    if theta_domain.upper is not None:
        rows.append(Constraint(add(target, constant(-theta_domain.upper)), '<='))
    reconstruction.append(Reconstruction(theta, target))
    objective = add(scale(target, objective_terms[0].coefficient), constant(model.objective.constant))
    reduced = replace(model, variables=tuple(variables), constraints=tuple(rows), objective=objective)
    return reduced, reconstruction, ('charnes_cooper', theta, str(lo), str(hi))


def reduce_model(model, *, max_terms=10000, fractional=True):
    """Try exact reductions to a fixed point; return a model and inverse map.

    Affine substitutions are accepted only if they decrease nonlinear degree
    burden without more than doubling total terms, or do not increase term
    count. The budget can skip opportunities, never relax constraints.
    """
    if not isinstance(max_terms, int) or isinstance(max_terms, bool) or max_terms < 1:
        raise ValueError('max_terms must be a positive integer')
    original = model
    reconstruction, steps = [], []
    while True:
        before = model
        model = eliminate_zero_variables(model).without_tautologies()
        new_zeros = set(model.eliminated_values) - set(before.eliminated_values)
        if new_zeros:
            steps.append(('zero_variables', tuple(sorted(new_zeros))))
        if fractional:
            converted = _fractional(model)
            if converted is not None:
                model, rules, step = converted
                reconstruction.extend(rules)
                steps.append(step)
        model, cancellation = _cancel_factors(model)
        steps.extend(cancellation)
        try:
            basis = _linear_basis(model, max_terms)
        except ExpansionLimit:
            steps.append(('skipped_affine_budget', max_terms))
            break
        if basis is None:
            steps.append(('skipped_inconsistent_exact_equalities',))
            break
        accepted = False
        old_cost = _cost(model)
        for name, replacement in sorted(basis, key=lambda item: (len(item[1].terms), item[0])):
            try:
                candidate = _eliminate(model, name, replacement, max_terms)
            except ExpansionLimit:
                continue
            cost = _cost(candidate)
            if ((cost[0] < old_cost[0] and cost[1] <= 2*max(1, old_cost[1]))
                    or (cost[0] <= old_cost[0] and cost[1] <= old_cost[1])
                    or (len(replacement.terms) <= 2 and cost[0] <= old_cost[0]
                        and cost[1] <= old_cost[1]+2)
                    or replacement.degree == 0):
                reconstruction.append(Reconstruction(name, replacement))
                steps.append(('affine_substitution', name, repr(replacement)))
                model = candidate
                accepted = True
                break
        if not accepted and not new_zeros and not cancellation:
            break
    return ReductionResult(original, model, tuple(reconstruction), tuple(steps))

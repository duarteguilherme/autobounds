"""Prove and eliminate zero-probability variables without choosing zero factors."""
from collections import Counter
from dataclasses import replace

from .model import Constraint


def _allows_zero(variable):
    return ((variable.lower is None or variable.lower <= 0)
            and (variable.upper is None or variable.upper >= 0))


def _monomial_sign(term, domains):
    """Return a guaranteed weak sign, or None when the sign is unknown."""
    sign = 1 if term.coefficient > 0 else -1
    unknown = False
    for name, power in Counter(term.variables).items():
        variable = domains[name]
        if variable.lower == variable.upper == 0:
            return 0
        if power % 2 == 0 or (variable.lower is not None and variable.lower >= 0):
            continue
        if variable.upper is not None and variable.upper <= 0:
            sign *= -1
        else:
            unknown = True
    return None if unknown else sign


def eliminate_zero_variables(model):
    """Project out provably zero decision variables and record reconstruction.

    Recognizes same-sign sums forced to zero, single zero monomials with only
    one possible zero factor, and consequences revealed by zero substitution.
    Domains and exact scalar equalities can establish nonzero factors. No
    tolerance is used, no parameter is removed, and ambiguous products remain.
    This is a sufficient-condition pass, not a complete assumption classifier.
    """
    domains = {variable.name: variable for variable in model.variables}
    eligible = {name for name, variable in domains.items()
                if variable.kind == 'decision' and _allows_zero(variable)}
    zeros = {name for name in eligible
             if domains[name].lower == domains[name].upper == 0}
    rows = list(model.constraints)
    while True:
        values = {name: 0 for name in zeros}
        rows = [Constraint(row.expression.substitute(values), row.relation)
                for row in rows]
        nonzero = {name for name, variable in domains.items()
                   if not _allows_zero(variable)}
        for row in rows:
            terms = [term for term in row.expression.terms if term.variables]
            if (row.relation == '==' and len(terms) == 1
                    and len(terms[0].variables) == 1):
                if row.expression.constant:
                    nonzero.add(terms[0].variables[0])
        inferred = set()
        for row in rows:
            expression = row.expression
            if expression.constant or not expression.terms:
                continue
            terms = expression.terms
            single_product = row.relation == '==' and len(terms) == 1
            if not single_product:
                signs = {_monomial_sign(term, domains) for term in terms}
                nonnegative = signs <= {0, 1}
                nonpositive = signs <= {0, -1}
                if not ((row.relation in ('==', '<=') and nonnegative)
                        or (row.relation in ('==', '>=') and nonpositive)):
                    continue
            for term in terms:
                # A zero product does not reveal which factor vanishes unless
                # all the other distinct factors are known to be nonzero.
                candidates = set(term.variables) - nonzero
                if len(candidates) == 1:
                    name = next(iter(candidates))
                    if name in eligible:
                        inferred.add(name)
        new = inferred - zeros
        if not new:
            break
        zeros.update(new)
    if not zeros:
        return model
    values = {name: 0 for name in zeros}
    removed = tuple(variable for variable in model.variables
                    if variable.name in zeros)
    return replace(
        model,
        variables=tuple(variable for variable in model.variables
                        if variable.name not in zeros),
        constraints=tuple(Constraint(row.expression.substitute(values), row.relation)
                          for row in rows),
        objective=model.objective.substitute(values),
        zero_variables=model.zero_variables + removed,
    ).without_tautologies()

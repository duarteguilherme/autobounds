"""Conservative polynomial rewrites with exact decimal coefficient arithmetic."""
from fractions import Fraction


from .numeric import decimal_token


def collect_terms(constraint, parameters, fixed=None):
    """Legacy adapter to the shared polynomial representation."""
    from .model import Polynomial
    fixed = fixed or {}
    expression = Polynomial.from_tokens(constraint[:-1], set(parameters) | set(fixed))
    expression = expression.substitute(fixed)
    return {term.variables: term.coefficient for term in expression.terms}


def write_constraint(terms, relation):
    result = []
    for variables, coefficient in sorted(terms.items()):
        coefficient_tokens = [] if coefficient == 1 and variables else [decimal_token(coefficient)]
        result.append(coefficient_tokens + list(variables))
    return result + [[relation]]


def simplify_model(model):
    """Return a new model after conservative, domain-preserving rewrites.

    Preconditions: explicit polynomial rows with ==, <= or >= relations.
    Only equalities with a single degree-one monomial can fix variables.
    Defining equations and domains survive; repeating decimal values are
    skipped so the legacy serializer need not approximate them.
    """
    from dataclasses import replace
    from .model import Constraint
    rows = list(model.constraints)
    fixed = {}
    definitions = []
    while True:
        remaining = []
        discovered = False
        for row in rows:
            expression = row.expression.substitute(fixed)
            variables = [term for term in expression.terms if term.variables]
            if (row.relation == '==' and len(variables) == 1
                    and len(variables[0].variables) == 1):
                variable = variables[0].variables[0]
                value = -expression.constant / variables[0].coefficient
                try:
                    decimal_token(value)
                except ValueError:
                    pass
                else:
                    fixed[variable] = value
                    definitions.append(Constraint(expression, row.relation))
                    discovered = True
                    continue
            remaining.append(Constraint(expression, row.relation))
        rows = remaining
        if not discovered:
            break
    constraints = tuple(dict.fromkeys(rows + definitions))
    return replace(model, constraints=constraints).without_tautologies(), fixed


def simplify_constraints(constraints, parameters):
    """Compatibility adapter; transformations operate on MathematicalModel."""
    from .model import MathematicalModel
    model = MathematicalModel.from_legacy(parameters, constraints)
    simplified, fixed = simplify_model(model)
    return simplified.legacy_constraints(), fixed

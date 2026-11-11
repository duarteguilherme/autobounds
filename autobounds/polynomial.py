"""Conservative polynomial rewrites with exact decimal coefficient arithmetic."""
from fractions import Fraction


def decimal_token(value):
    """Serialize a terminating rational exactly; reject repeating decimals."""
    value = Fraction(value)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError('Coefficient does not have a finite decimal expansion')
    places = max(twos, fives)
    numerator = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    sign = '-' if numerator < 0 else ''
    digits = str(abs(numerator)).zfill(places + 1)
    if not places:
        return sign + digits
    return (sign + digits[:-places] + '.' + digits[-places:]).rstrip('0').rstrip('.')


def collect_terms(constraint, parameters, fixed=None):
    """Combine identical monomials, including every repeated variable factor."""
    fixed = fixed or {}
    result = {}
    for term in constraint[:-1]:
        coefficient = Fraction(1)
        variables = []
        for token in term:
            if token in fixed:
                coefficient *= fixed[token]
            elif token in parameters:
                variables.append(token)
            else:
                coefficient *= Fraction(str(token))
        monomial = tuple(sorted(variables))
        result[monomial] = result.get(monomial, Fraction(0)) + coefficient
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def write_constraint(terms, relation):
    result = []
    for variables, coefficient in sorted(terms.items()):
        coefficient_tokens = [] if coefficient == 1 and variables else [decimal_token(coefficient)]
        result.append(coefficient_tokens + list(variables))
    return result + [[relation]]


def simplify_constraints(constraints, parameters):
    """Collect terms and propagate fixed variables, retaining defining equalities.

    No inequalities are used as equalities, no variables/domains are deleted,
    and no division by a model variable is performed. Nonterminating fixed
    values are skipped to avoid introducing approximate coefficients.
    """
    parameters = set(parameters)
    rows = []
    for constraint in constraints:
        if not constraint:
            continue
        relation = constraint[-1][0]
        if relation not in ('==', '<=', '>='):
            raise ValueError(f'Unsupported constraint relation: {relation}')
        rows.append((collect_terms(constraint, parameters), relation))
    fixed = {}
    definitions = []
    while True:
        remaining = []
        discovered = False
        for terms, relation in rows:
            row = write_constraint(terms, relation)
            terms = collect_terms(row, parameters, fixed)
            variables = [monomial for monomial in terms if monomial]
            if relation == '==' and len(variables) == 1 and len(variables[0]) == 1:
                variable = variables[0][0]
                value = -terms.get((), Fraction(0)) / terms[variables[0]]
                try:
                    decimal_token(value)
                except ValueError:
                    pass
                else:
                    fixed[variable] = value
                    # Keep the defining equation, including its original scale.
                    definitions.append(write_constraint(terms, relation))
                    discovered = True
                    continue
            remaining.append((terms, relation))
        rows = remaining
        if not discovered:
            break
    rewritten = [write_constraint(terms, relation) for terms, relation in rows]
    rewritten.extend(definitions)
    unique = []
    seen = set()
    for row in rewritten:
        key = tuple(tuple(term) for term in row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique, fixed

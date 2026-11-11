"""Immutable, solver-independent polynomial models with exact coefficients."""
from dataclasses import dataclass, replace
from fractions import Fraction


def rational(value):
    """Interpret decimal inputs by their written value, not binary expansion."""
    return value if isinstance(value, Fraction) else Fraction(str(value))


@dataclass(frozen=True)
class Variable:
    name: str
    lower: Fraction | None = Fraction(0)
    upper: Fraction | None = Fraction(1)
    kind: str = 'decision'
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Variables require nonempty string names')
        if self.kind not in ('decision', 'parameter'):
            raise ValueError('Variable kind must be decision or parameter')
        for field in ('lower', 'upper'):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, rational(value))
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError('Variable lower bound exceeds upper bound')


@dataclass(frozen=True)
class Term:
    coefficient: Fraction
    variables: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'coefficient', rational(self.coefficient))
        object.__setattr__(self, 'variables', tuple(sorted(self.variables)))


@dataclass(frozen=True)
class Polynomial:
    terms: tuple[Term, ...] = ()

    def __post_init__(self):
        coefficients = {}
        for term in self.terms:
            coefficients[term.variables] = coefficients.get(term.variables, Fraction(0)) + term.coefficient
        object.__setattr__(self, 'terms', tuple(
            Term(coefficient, variables) for variables, coefficient in sorted(coefficients.items())
            if coefficient))

    @classmethod
    def from_tokens(cls, terms, variables):
        names = set(variables)
        result = []
        for factors in terms:
            coefficient = Fraction(1)
            monomial = []
            for factor in factors:
                if factor in names:
                    monomial.append(factor)
                else:
                    try:
                        coefficient *= rational(factor)
                    except (ValueError, ZeroDivisionError) as exc:
                        raise ValueError(f'Unknown variable or invalid coefficient: {factor!r}') from exc
            result.append(Term(coefficient, tuple(monomial)))
        return cls(tuple(result))

    @property
    def degree(self):
        return max((len(term.variables) for term in self.terms), default=0)

    @property
    def constant(self):
        return sum((term.coefficient for term in self.terms if not term.variables), Fraction(0))

    def substitute(self, values):
        values = {name: rational(value) for name, value in values.items()}
        result = []
        for term in self.terms:
            coefficient = term.coefficient
            variables = []
            for name in term.variables:
                if name in values:
                    coefficient *= values[name]
                else:
                    variables.append(name)
            result.append(Term(coefficient, tuple(variables)))
        return Polynomial(tuple(result))

    def evaluate(self, values):
        result = self.substitute(values)
        if result.degree:
            raise ValueError('Values are missing for polynomial variables')
        return result.constant


@dataclass(frozen=True)
class Constraint:
    """A polynomial compared with zero; RHS terms belong in expression."""
    expression: Polynomial
    relation: str = '=='

    def __post_init__(self):
        if self.relation not in ('==', '<=', '>='):
            raise ValueError(f'Unsupported constraint relation: {self.relation}')

    @property
    def is_tautology(self):
        if self.expression.degree:
            return False
        value = self.expression.constant
        return {'==': value == 0, '<=': value <= 0, '>=': value >= 0}[self.relation]


@dataclass(frozen=True)
class MathematicalModel:
    variables: tuple[Variable, ...]
    constraints: tuple[Constraint, ...]
    objective: Polynomial
    zero_variables: tuple[Variable, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'variables', tuple(self.variables))
        object.__setattr__(self, 'constraints', tuple(self.constraints))
        object.__setattr__(self, 'zero_variables', tuple(self.zero_variables))
        names = [variable.name for variable in self.variables]
        if len(set(names)) != len(names):
            raise ValueError('Variable names must be unique')
        eliminated_names = [variable.name for variable in self.zero_variables]
        if len(set(eliminated_names)) != len(eliminated_names) or set(names) & set(eliminated_names):
            raise ValueError('Eliminated variable names must be unique and absent from the reduced model')
        for variable in self.zero_variables:
            if (variable.kind != 'decision' or
                    (variable.lower is not None and variable.lower > 0) or
                    (variable.upper is not None and variable.upper < 0)):
                raise ValueError('Eliminated zero variables must permit zero in their domain')
        declared = set(names)
        for expression in (self.objective, *(row.expression for row in self.constraints)):
            used = {name for term in expression.terms for name in term.variables}
            if used - declared:
                raise ValueError(f'Undeclared variables: {sorted(used - declared)}')

    @property
    def eliminated_values(self):
        """Exact zeros for restoring eliminated decision variables."""
        return {variable.name: Fraction(0) for variable in self.zero_variables}

    def restore_solution(self, values):
        """Return a fresh solution dictionary including proven zero variables."""
        return {**values, **{variable.name: 0.0 for variable in self.zero_variables}}

    @property
    def is_linear(self):
        return self.objective.degree <= 1 and all(row.expression.degree <= 1 for row in self.constraints)

    @property
    def parameters(self):
        return tuple(variable for variable in self.variables if variable.kind == 'parameter')

    def bind(self, values):
        """Bind symbolic data by internal name, preserving all decision domains."""
        required = {variable.name for variable in self.parameters}
        if set(values) != required:
            raise ValueError(f'Provide exactly these symbolic parameters: {sorted(required)}')
        for variable in self.parameters:
            value = rational(values[variable.name])
            if ((variable.lower is not None and value < variable.lower)
                    or (variable.upper is not None and value > variable.upper)):
                raise ValueError(f'Parameter outside its domain: {variable.name}')
        return MathematicalModel(
            tuple(variable for variable in self.variables if variable.kind == 'decision'),
            tuple(Constraint(row.expression.substitute(values), row.relation) for row in self.constraints),
            self.objective.substitute(values),
            self.zero_variables,
        )

    def without_tautologies(self):
        return replace(self, constraints=tuple(row for row in self.constraints if not row.is_tautology))

    @classmethod
    def from_legacy(cls, parameters, constraints, limits=(None, None)):
        variables = tuple(Variable(name, *limits) if name == 'objvar' else Variable(name)
                          for name in parameters)
        rows = tuple(Constraint(Polynomial.from_tokens(row[:-1], parameters), row[-1][0])
                     for row in constraints if row)
        objective = Polynomial((Term(1, ('objvar',)),)) if 'objvar' in parameters else Polynomial()
        return cls(variables, rows, objective)

    def legacy_constraints(self):
        """Return fresh legacy rows; rationals remain exact if not terminating."""
        from .numeric import decimal_token
        rows = []
        for constraint in self.constraints:
            terms = []
            for term in constraint.expression.terms:
                try:
                    coefficient = decimal_token(term.coefficient)
                except ValueError:
                    coefficient = term.coefficient
                factors = [] if term.coefficient == 1 and term.variables else [coefficient]
                terms.append(factors + list(term.variables))
            rows.append(terms + [[constraint.relation]])
        return rows

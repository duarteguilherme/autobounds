"""Graph-independent declaration of linear response-distribution models."""
import sympy as sp
from .symbolic_results import SymbolicBounds


class LinearResponseModel:
    """The exact model q >= 0, sum(q) = 1, M q = p, target c' q.

    Rows of ``matrix`` are observable event indicators. ``target`` accepts
    signed rational coefficients. Every distribution over the supplied
    response types must be admissible apart from these linear equations.
    Optional ``constraints`` are additional signed rational rows with
    symbolic right-hand sides named by ``constraint_labels``. They encode
    equalities, not inequalities. ``matrix`` and ``labels`` on the compiled
    model include both data and constraint rows. No independence constraints
    on q are implicit.
    """

    def __init__(self, matrix, target, labels=None, response_types=None, *,
                 constraints=(), constraint_labels=None):
        self.target = tuple(sp.Rational(value) for value in target)
        if not self.target:
            raise ValueError("At least one admissible response type is required.")
        self.matrix = tuple(tuple(sp.Rational(value) for value in row) for row in matrix)
        if any(len(row) != len(self.target) for row in self.matrix):
            raise ValueError("Every incidence row must match the target length.")
        if any(value not in (0, 1) for row in self.matrix for value in row):
            raise ValueError("Data rows must be event indicators (zero or one).")
        if labels is None:
            labels = [f"p{i}" for i in range(len(self.matrix))]
        self.labels = tuple(map(str, labels))
        if len(self.labels) != len(self.matrix) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Provide one distinct label per data row.")
        constraints = tuple(tuple(sp.Rational(value) for value in row)
                            for row in constraints)
        if any(len(row) != len(self.target) for row in constraints):
            raise ValueError("Every constraint row must match the target length.")
        if any(not value.is_finite for row in constraints for value in row):
            raise ValueError("Constraint coefficients must be finite rationals.")
        if constraint_labels is None:
            constraint_labels = [f"t{i}" for i in range(len(constraints))]
        constraint_labels = tuple(map(str, constraint_labels))
        if (len(constraint_labels) != len(constraints)
                or any(not label.strip() for label in constraint_labels)
                or len(set(self.labels + constraint_labels))
                != len(self.labels) + len(constraint_labels)):
            raise ValueError("Provide one distinct label per data or constraint row.")
        self.matrix += constraints
        self.labels += constraint_labels
        self.symbols = tuple(sp.Symbol(label, real=True) for label in self.labels)
        self.response_types = tuple(range(len(self.target)) if response_types is None
                                    else response_types)
        if len(self.response_types) != len(self.target):
            raise ValueError("Response type labels must match the target length.")

    def compile(self):
        """Compile into the shared polynomial model with explicit data parameters."""
        from .model import Constraint, MathematicalModel, Polynomial, Term, Variable
        weights = tuple(Variable(f'q{i}', label=str(label))
                        for i, label in enumerate(self.response_types))
        data = tuple(Variable(f'p{i}', None, None, 'parameter', label)
                     for i, label in enumerate(self.labels))
        normalization = Constraint(Polynomial(
            tuple(Term(1, (variable.name,)) for variable in weights) + (Term(-1),)))
        rows = [normalization]
        for values, parameter in zip(self.matrix, data):
            terms = tuple(Term(value, (variable.name,)) for value, variable in zip(values, weights))
            rows.append(Constraint(Polynomial(terms + (Term(-1, (parameter.name,)),))))
        objective = Polynomial(tuple(Term(value, (variable.name,))
                                     for value, variable in zip(self.target, weights)))
        return MathematicalModel(weights + data, tuple(rows), objective)

    def solve(self, *, linearize=False):
        """Solve the compiled model with the exact response-mixture backend."""
        from .backends.cdd import CddBackend
        if linearize:
            return self.compile().linearize().solve('cdd')
        return CddBackend().solve(self.compile())

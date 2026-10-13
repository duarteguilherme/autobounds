"""Graph-based declarations for exact symbolic linear response problems."""

from copy import deepcopy
from itertools import product
from numbers import Integral
import re

import sympy as sp

from .Parser import Parser
from .Q import Q
from .linear_response import LinearResponseModel


class SymbolicProblem:
    """Declare a graph, symbolic data, and a linear causal estimand.

    Like Bounder, ``p`` returns queries supporting arithmetic. Unlike Bounder,
    ``load_data`` declares available distributions, not their numerical values.
    ``solve`` returns sharp symbolic lower and upper envelopes.

    The initial graph adapter supports queries and data confined to one
    unrestricted response component. A nonsingleton component must have a
    latent parent shared by all its variables. Separate independent components
    are never pooled. Conditioning on independent observed roots is supported
    via its equivalent intervention; conditioning elsewhere is rejected.
    """

    def __init__(self, dag, number_values=None):
        self.dag = deepcopy(dag)
        supplied = {} if number_values is None else dict(number_values)
        if set(supplied) - self.dag.V:
            raise ValueError("Cardinalities must refer to observed graph variables.")
        self.number_values = {v: supplied.get(v, 2) for v in self.dag.V}
        if any(not isinstance(n, Integral) or isinstance(n, bool) or not 1 <= n <= 10
               for n in self.number_values.values()):
            raise ValueError("The canonical parser supports cardinalities from 1 to 10.")
        if not self.dag.V:
            raise ValueError("The graph must contain observed variables.")
        if any(self.dag.find_parents(u) for u in self.dag.U):
            raise ValueError("Only graphs with root latent variables are supported.")
        # Take the transitive closure of shared-latent links before invoking
        # the canonical parser, whose component finder assumes those links
        # already span each component in one step.
        remaining = set(self.dag.V)
        while remaining:
            component = set()
            pending = {min(remaining)}
            while pending:
                variable = pending.pop()
                component.add(variable)
                for latent in self.dag.find_parents_u(variable):
                    pending.update((self.dag.find_children(latent) & self.dag.V) - component)
            remaining -= component
            if len(component) > 1:
                common = set.intersection(*(self.dag.find_parents_u(v) for v in component))
                if not common:
                    raise ValueError("A response component without a common latent parent "
                                     "can impose nonlinear restrictions; it is not supported.")
        self.Parser = Parser(self.dag, dict(self.number_values))
        self.canModel = self.Parser.canModel
        self._components = tuple(tuple(c) for c in self.Parser.c_parameters)
        self._symbols = {p: sp.Symbol(p) for c in self._components for p in c}
        self._component_of = {self._symbols[p]: i for i, c in enumerate(self._components)
                              for p in c}
        self._normalization = {self._symbols[c[-1]]:
                               1 - sum(self._symbols[p] for p in c[:-1])
                               for c in self._components}
        self._data = {}
        self._zero_events = []
        self.estimand = None

    def _variables(self, variables):
        if isinstance(variables, str):
            variables = variables.split(',')
        variables = tuple(v.strip() for v in variables)
        if not variables or len(set(variables)) != len(variables):
            raise ValueError("Provide a nonempty list of distinct variable names.")
        if set(variables) - self.dag.V:
            raise ValueError(f"Unknown observed variables: {set(variables) - self.dag.V}")
        return variables

    def _assignments(self, text):
        assignments = {}
        for term in text.replace(' ', '').replace('&', ',').split(','):
            match = re.fullmatch(r'([^=(),&]+)=(\d+)', term)
            if match is None:
                raise ValueError(f"Invalid assignment: {term!r}")
            variable, value = match[1], int(match[2])
            if variable not in self.number_values or value >= self.number_values[variable]:
                raise ValueError(f"Unknown variable or out-of-range value: {term}")
            if variable in assignments and assignments[variable] != value:
                raise ValueError(f"Conflicting assignments for {variable}.")
            assignments[variable] = value
        return assignments

    def _root_condition(self, condition):
        assignments = self._assignments(condition)
        if any(self.dag.find_parents(v) for v in assignments):
            raise ValueError("Conditioning is supported only on independent observed roots; "
                             "use do= for genuinely interventional data.")
        return assignments

    @staticmethod
    def _assignment_text(assignments):
        return ','.join(f'{v}={value}' for v, value in assignments.items())

    def p(self, event, cond=None):
        """Return a Bounder-style Q query; signed linear arithmetic is allowed.

        Counterfactual syntax is e.g. ``Y(X=1)=1``. An optional condition
        must assign independent root variables and the event must be factual.
        """
        if not isinstance(event, str):
            raise TypeError("Events must be strings.")
        event = event.replace(' ', '')
        if event.startswith('P(') and event.endswith(')'):
            event = event[2:-1]
        atoms = []
        for atom in event.split('&'):
            match = re.fullmatch(r'([^=(),&]+)(?:\(([^()]+)\))?=(\d+)', atom)
            if match is None:
                raise ValueError(f"Invalid event: {atom!r}")
            variable, world, value = match[1], match[2], int(match[3])
            self._assignments(f'{variable}={value}')
            assignments = {} if world is None else self._assignments(world)
            if cond is not None:
                if world is not None:
                    raise ValueError("Use explicit interventions for conditioned counterfactuals.")
                assignments.update(self._root_condition(cond))
            if variable in assignments:
                if assignments[variable] != value:
                    return Q(0)
                continue
            suffix = f'({self._assignment_text(assignments)})' if assignments else ''
            atoms.append(f'{variable}{suffix}={value}')
        if not atoms:
            return Q(1)
        return self.Parser.p('&'.join(dict.fromkeys(atoms)))

    def _linear_expression(self, query):
        if not isinstance(query, Q):
            query = Q(query)
        if query._cond is not None:
            raise ValueError("Conditional ratios are not linear response estimands.")
        expression = sp.Integer(0)
        for coefficient, factors in query._event:
            term = sp.Rational(coefficient)
            for factor in factors:
                if str(factor) == '1':
                    continue
                if factor not in self._symbols:
                    raise ValueError(f"Unknown response parameter: {factor}")
                term *= self._symbols[factor]
            expression += term
        expression = sp.expand(expression.xreplace(self._normalization))
        if expression.free_symbols and sp.Poly(expression).total_degree() > 1:
            raise ValueError("This query is nonlinear in the response probabilities.")
        return expression

    def set_estimand(self, estimand):
        """Set a rational linear query, including differences such as an ATE."""
        query = self.p(estimand) if isinstance(estimand, str) else estimand
        self.estimand = self._linear_expression(query)
        return self

    def set_ate(self, ind, dep):
        """Set P(dep(ind=1)=1) - P(dep(ind=0)=1)."""
        return self.set_estimand(self.p(f'{dep}({ind}=1)=1') -
                                 self.p(f'{dep}({ind}=0)=1'))

    def add_data(self, event, label=None):
        """Declare one available event probability, optionally naming its symbol."""
        if label is None:
            if not isinstance(event, str):
                raise ValueError("Supply a symbolic label when declaring data using Q.")
            label = f'P({event})'
        label = str(label)
        if not label:
            raise ValueError("Data labels cannot be empty.")
        query = self.p(event) if isinstance(event, str) else event
        expression = self._linear_expression(query)
        if label in self._data and self._data[label] != expression:
            raise ValueError(f"Label {label!r} already denotes a different event.")
        self._data[label] = expression
        return self

    def load_data(self, variables, cond=None, do=None):
        """Declare every cell of a symbolic distribution.

        ``load_data('X,Y', cond='Z')`` declares P(X,Y | Z=z) for every
        instrument value. ``load_data('Y', do='X=0')`` declares experimental
        outcome cells. Numeric tables are not accepted. Root conditional
        cells assume that the population conditional distributions exist.
        """
        variables = self._variables(variables)
        if cond is not None and do is not None:
            raise ValueError("Declare conditional and interventional data separately.")
        conditions = self._variables(cond) if cond is not None else ()
        interventions = self._assignments(do) if do is not None else {}
        if set(variables) & (set(conditions) | set(interventions)):
            raise ValueError("Data variables must exclude conditioning/intervention variables.")
        for variable in conditions:
            self._root_condition(f'{variable}=0')
        pending = []
        for condition_values in product(*(range(self.number_values[v]) for v in conditions)):
            condition = self._assignment_text(dict(zip(conditions, condition_values)))
            for values in product(*(range(self.number_values[v]) for v in variables)):
                event = '&'.join(f'{v}={value}' for v, value in zip(variables, values))
                if conditions:
                    query = self.p(event, cond=condition)
                    label = f'P({event}|{condition})'
                elif interventions:
                    world = self._assignment_text(interventions)
                    query = self.p('&'.join(f'{v}({world})={value}'
                                            for v, value in zip(variables, values)))
                    label = f'P({event}|do({world}))'
                else:
                    query, label = self.p(event), f'P({event})'
                pending.append((label, self._linear_expression(query)))
        for label, expression in pending:
            if label in self._data and self._data[label] != expression:
                raise ValueError(f"Label {label!r} already denotes a different event.")
        self._data.update(pending)
        return self

    def set_p_to_zero(self, event):
        """Exclude response types in an impossible event, e.g. IV defiers."""
        query = self.p(event) if isinstance(event, str) else event
        self._zero_events.append(self._linear_expression(query))
        return self

    def write_program(self):
        """Compile declarations into an explicit LinearResponseModel."""
        if self.estimand is None:
            raise ValueError("Set an estimand before generating the problem.")
        expressions = [self.estimand] + list(self._data.values()) + self._zero_events
        active = {self._component_of[s] for expr in expressions for s in expr.free_symbols}
        if len(active) > 1:
            raise ValueError("Declarations involve multiple independent response components. "
                             "This adapter currently supports one unrestricted component; "
                             "it will not pool independent latent distributions.")
        component = self._components[next(iter(active))] if active else self._components[0]
        symbols = [self._symbols[p] for p in component]

        def vector(expression):
            constant = expression.xreplace({symbol: 0 for symbol in expression.free_symbols})
            return tuple(constant + expression.coeff(symbol) for symbol in symbols)
        target = vector(self.estimand)
        matrix = [vector(expr) for expr in self._data.values()]
        zero_events = [vector(expr) for expr in self._zero_events]
        if any(value not in (0, 1) for row in matrix + zero_events for value in row):
            raise ValueError("Data declarations and zero restrictions must be event indicators.")
        retained = [i for i in range(len(component)) if all(row[i] == 0 for row in zero_events)]
        return LinearResponseModel(
            [[row[i] for i in retained] for row in matrix],
            [target[i] for i in retained], labels=self._data,
            response_types=[component[i] for i in retained])

    def solve(self):
        """Return complete sharp bounds using the exact GMP/cdd hull solver."""
        return self.write_program().solve()

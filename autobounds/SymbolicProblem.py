"""Graph-based declarations for exact symbolic linear response problems."""

from copy import deepcopy
from itertools import product
from numbers import Integral
import re

import sympy as sp

from .Parser import Parser
from .Q import Q
from .exceptions import UnsupportedModelError
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

    Solver selection is automatic. Simple linear declarations use the fast
    response solver. When reduction is needed, independent components are bound from
    symbolic data, and a response-feature projection automatically reduces
    polynomial declarations to a parametric LP when possible. Homogeneity
    can be declared using event pairs and a partition of response strata.
    Unsupported nonlinearities and unknown elimination signs raise errors.
    Explicit ``parametric=True`` or ``False`` overrides automatic selection.
    ``solver_selection`` records the selected method and reason. Unexpected
    backend errors propagate without retries or fallback to another solver.
    """

    def __init__(self, dag, number_values=None, *, parametric=None):
        if parametric is not None and type(parametric) is not bool:
            raise TypeError("parametric must be None (automatic), True, or False")
        self.parametric = parametric
        self.solver_selection = None
        self._homogeneity = []
        self._assumptions = []
        self._positive_roots = []
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
            raise UnsupportedModelError("Only graphs with root latent variables are supported.")
        # Check every connected latent component: the symbolic linear backend
        # cannot represent within-component independence restrictions.
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
                    raise UnsupportedModelError("A response component without a common latent parent "
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
        self._constraints = {}
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
            raise UnsupportedModelError("Conditioning is supported only on independent observed roots; "
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

    def _linear_expression(self, query, *, normalize=True):
        if not isinstance(query, Q):
            query = Q(query)
        if query._cond is not None:
            raise UnsupportedModelError("Conditional ratios are not linear response estimands.")
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
        if self.parametric is not False or not normalize:
            return sp.expand(expression)
        expression = sp.expand(expression.xreplace(self._normalization))
        if expression.free_symbols and sp.Poly(expression).total_degree() > 1:
            raise UnsupportedModelError("This query is nonlinear in the response probabilities.")
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
        if label in self._constraints:
            raise ValueError(f"Label {label!r} already denotes a constraint.")
        if label in self._data and self._data[label] != expression:
            raise ValueError(f"Label {label!r} already denotes a different event.")
        self._data[label] = expression
        return self

    def add_constraint(self, expression, label):
        """Declare a linear equality ``expression = label``.

        ``expression`` is a Q query (including signed rational contrasts),
        an event string, or a rational constant. ``label`` names a symbolic
        right-hand side, not a formula to parse. For example::

            problem.add_constraint(numerator0, label="t")
            problem.set_estimand(numerator1)
            bounds = problem.solve()

        Substitute a previously computed bound for t in the resulting
        envelopes AND compatibility conditions. Only compatible values of t
        describe feasible distributions. Piecewise bounds require their
        corresponding parameter regions. Inequalities, nonlinear queries,
        and symbolic coefficients of response probabilities are unsupported.
        """
        label = str(label)
        if not label.strip():
            raise ValueError("Constraint labels cannot be empty.")
        if label in self._data:
            raise ValueError(f"Label {label!r} already denotes data.")
        query = self.p(expression) if isinstance(expression, str) else expression
        expression = self._linear_expression(query)
        if label in self._constraints and self._constraints[label] != expression:
            raise ValueError(f"Label {label!r} already denotes a different constraint.")
        self._constraints[label] = expression
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
            if self.parametric is not False:
                for value in range(self.number_values[variable]):
                    self._positive_roots.append(self._linear_expression(self.p(f'{variable}={value}')))
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
            if label in self._constraints:
                raise ValueError(f"Label {label!r} already denotes a constraint.")
            if label in self._data and self._data[label] != expression:
                raise ValueError(f"Label {label!r} already denotes a different event.")
        self._data.update(pending)
        return self

    def add_assumption(self, constraint, symbol='==', constraint2=None):
        """Declare assumptions; automatically remove provably excluded types.

        An event string or Q query defaults to equality with zero. ``symbol``
        and ``constraint2`` specify a general probability equality/inequality,
        matching Bounder's convention. Only sign-definite affine restrictions
        that force response probabilities to zero use ``set_p_to_zero``.
        Signed contrasts and nonlinear products remain population constraints.

        Use ``add_homogeneity`` separately for equality of average effects.
        """
        if constraint is None:
            raise ValueError('Supply an assumption')
        if symbol not in ('==', '<=', '>='):
            raise ValueError('Assumption relation must be ==, <=, or >=')

        def expression(value):
            if not isinstance(value, (Q, str)):
                return sp.Rational(str(value))
            query = self.p(value) if isinstance(value, str) else value
            return self._linear_expression(query, normalize=False)

        e = sp.expand(expression(constraint)-expression(0 if constraint2 is None else constraint2))
        if not e.free_symbols:
            valid = {'==': e == 0, '<=': e <= 0, '>=': e >= 0}[symbol]
            if not valid:
                raise ValueError('Assumption is contradictory')
            return self
        if sp.Poly(e).total_degree() <= 1:
            active = {self._component_of[v] for v in e.free_symbols}
            constant = e.subs({v: 0 for v in e.free_symbols})
            if len(active) == 1:
                component = self._components[next(iter(active))]
                # sum(q)=1 turns probability-one assumptions into exclusions
                # of their complements, without dividing by a stratum mass.
                coefficients = {self._symbols[name]: e.coeff(self._symbols[name])+constant
                                for name in component}
            elif constant == 0:
                coefficients = {v: e.coeff(v) for v in e.free_symbols}
            else:
                coefficients = {}
            if coefficients:
                nonnegative = all(v >= 0 for v in coefficients.values())
                nonpositive = all(v <= 0 for v in coefficients.values())
                if ((nonnegative and symbol == '>=') or
                        (nonpositive and symbol == '<=')):
                    return self
                force_zero = ((nonnegative and symbol in ('==', '<=')) or
                              (nonpositive and symbol in ('==', '>=')))
                if force_zero:
                    excluded = [v for v, coefficient in coefficients.items() if coefficient != 0]
                    if excluded:
                        self.set_p_to_zero(Q([(1, [str(v)]) for v in excluded]))
                    return self
        if self.parametric is False:
            raise UnsupportedModelError(
                'Population assumptions are unsupported with forced parametric=False; '
                'omit the flag for automatic selection')
        self._assumptions.append((e, symbol))
        return self

    def set_p_to_zero(self, event):
        """Low-level zero restriction; public declarations use add_assumption."""
        query = self.p(event) if isinstance(event, str) else event
        self._zero_events.append(self._linear_expression(query))
        return self

    def add_homogeneity(self, treated, control, *, strata):
        """Equalize E[1(treated)-1(control)|stratum] across a partition.

        Supply two event strings and principal-stratum event strings. This
        declares numerator_s = mass_s * population_effect, including the
        vacuous zero-mass case. The compiler verifies the partition after
        applying zero-event restrictions. Automatic selection enables reduction.
        """
        if self.parametric is False:
            raise UnsupportedModelError('Homogeneity is unsupported with forced parametric=False; omit the flag for automatic selection')
        if not isinstance(treated, str) or not isinstance(control, str):
            raise TypeError('Homogeneity outcomes must be event strings')
        if strata is None or isinstance(strata, str):
            raise ValueError('Supply a nonempty sequence of stratum events')
        strata = tuple(strata)
        if not strata or any(not isinstance(s, str) for s in strata):
            raise ValueError('Supply a nonempty sequence of stratum events')
        effect = self._linear_expression(self.p(treated)-self.p(control))
        masses = tuple(self._linear_expression(self.p(s)) for s in strata)
        deltas = tuple(self._linear_expression(self.p(treated+'&'+s)-self.p(control+'&'+s))
                       for s in strata)
        self._homogeneity.append((effect, masses, deltas))
        return self

    def write_program(self):
        """Compile declarations into a linear or parametric response program."""
        if self.estimand is None:
            raise ValueError("Set an estimand before generating the problem.")
        self.solver_selection = None
        if self.parametric is False and self._assumptions:
            raise UnsupportedModelError(
                'Population assumptions are unsupported with forced parametric=False; '
                'omit the flag for automatic selection')
        if self.parametric is False and self._homogeneity:
            raise UnsupportedModelError('Homogeneity is unsupported with forced parametric=False; omit the flag for automatic selection')
        raw = ([self.estimand] + list(self._data.values())
               + list(self._constraints.values()) + self._zero_events)
        normalized = [sp.expand(e.xreplace(self._normalization)) for e in raw]
        nonlinear = any(e.free_symbols and sp.Poly(e).total_degree() > 1
                        for e in normalized)
        active = {self._component_of[s] for e in normalized for s in e.free_symbols}
        reason = ('homogeneity assumptions' if self._homogeneity else
                  'population assumptions' if self._assumptions else
                  'nonlinear response expressions' if nonlinear else
                  'multiple independent response components' if len(active) > 1 else None)
        use_reduction = self.parametric is True or (self.parametric is None and reason is not None)
        if use_reduction:
            from .parametric import ParametricResponseProgram
            self.solver_selection = dict(method='parametric_reduction',
                                         reason=reason or 'explicit override')
            return ParametricResponseProgram(
                tuple(tuple(self._symbols[p] for p in c) for c in self._components),
                dict(self._data), dict(self._constraints), tuple(self._zero_events),
                self.estimand, tuple(self._homogeneity), tuple(dict.fromkeys(self._positive_roots)),
                assumptions=tuple(self._assumptions))
        if nonlinear:
            raise UnsupportedModelError('This query is nonlinear in the response probabilities.')
        if len(active) > 1:
            raise UnsupportedModelError("Declarations involve multiple independent response components. "
                             "The forced linear adapter supports one unrestricted component; "
                             "omit parametric=False for automatic selection.")
        self.solver_selection = dict(method='linear_response',
                                     reason='fixed-coefficient linear response model')
        estimand = normalized[0]
        data_end = 1 + len(self._data)
        constraint_end = data_end + len(self._constraints)
        data = normalized[1:data_end]
        constraints = normalized[data_end:constraint_end]
        zeros = normalized[constraint_end:]
        component = self._components[next(iter(active))] if active else self._components[0]
        symbols = [self._symbols[p] for p in component]

        def vector(expression):
            constant = expression.xreplace({symbol: 0 for symbol in expression.free_symbols})
            return tuple(constant + expression.coeff(symbol) for symbol in symbols)
        target = vector(estimand)
        matrix = [vector(expr) for expr in data]
        constraints = [vector(expr) for expr in constraints]
        zero_events = [vector(expr) for expr in zeros]
        if any(value not in (0, 1) for row in matrix + zero_events for value in row):
            raise ValueError("Data declarations and zero restrictions must be event indicators.")
        retained = [i for i in range(len(component)) if all(row[i] == 0 for row in zero_events)]
        return LinearResponseModel(
            [[row[i] for i in retained] for row in matrix],
            [target[i] for i in retained], labels=self._data,
            response_types=[component[i] for i in retained],
            constraints=[[row[i] for i in retained] for row in constraints],
            constraint_labels=self._constraints)

    def compile(self):
        """Return the shared model (automatically reduced in parametric mode)."""
        return self.write_program().compile()

    def solve(self, *, linearize=False):
        """Return exact symbolic bounds; parametric mode derives its reduction."""
        program = self.write_program()
        if isinstance(program, LinearResponseModel):
            return program.solve(linearize=linearize)
        return program.solve()

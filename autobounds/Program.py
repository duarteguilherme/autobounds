from fractions import Fraction
import random
from .ProgramUtils import (
    change_constraint_parameter_value, parse_whole_file_scip, plot_bounds,
)
import os


alphanum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

class Program:
    """Polynomial constraints and an objective for SCIP or PIP export.

    Constraint rows are sums of products compared with zero. Ordinary
    parameters have probability bounds [0, 1]; objvar carries the objective.
    """
    def __init__(self):
        self._zero_variables = ()
        self.parameters = [ ]
        self.constraints = [ tuple() ]
        self.res_scip = None
        self._scip_bounds = None
        self.last_result = None
        self.scip_lower_filename = ''.join(random.choices(alphanum, k = 16)) + '.log2n'
        self.scip_upper_filename = ''.join(random.choices(alphanum, k = 16)) + '.log2n'

    def __del__(self):
        try:
            if os.path.exists(self.scip_lower_filename):
                os.remove(self.scip_lower_filename)
            if os.path.exists(self.scip_upper_filename):
                os.remove(self.scip_upper_filename)
        except Exception:
            pass

    def simplify(self):
        """Simplify in place while retaining fixed-variable equations and domains.

        This opt-in pass collects polynomial terms, removes exact duplicate
        rows and propagates single-variable equalities with finite decimal
        solutions. It does not perform general linear elimination.
        """
        from .polynomial import simplify_model
        before_constraints = len(self.constraints)
        before_terms = sum(max(0, len(row) - 1) for row in self.constraints)
        model, fixed = simplify_model(self.compile())
        self.parameters = [variable.name for variable in model.variables]
        self._zero_variables = model.zero_variables
        self.constraints = model.legacy_constraints()
        self._scip_bounds = None
        self.last_result = None
        return {
            "constraints_before": before_constraints,
            "constraints_after": len(self.constraints),
            "terms_before": before_terms,
            "terms_after": sum(len(row) - 1 for row in self.constraints),
            "fixed_parameters": tuple(sorted(fixed)),
            "eliminated_parameters": tuple(model.eliminated_values),
        }

    def plot(self):
        self.track_result_scip()
        self.res_scip = self.res_scip.loc[lambda k: k.seconds != 'time']
        self.res_scip = self.res_scip.astype(float)
        self.res_scip = self.res_scip.sort_values(['seconds'])
        self.res_scip = self.res_scip.fillna(method = 'ffill')
        self.res_scip = self.res_scip.fillna(method = 'bfill')
        return plot_bounds(self.res_scip)

    def linearize(self, limits=(None, None), **options):
        """Return an exact reduction with a report and solution reconstruction.

        Leaves these mutable declarations untouched. Use result.solve('cdd')
        for exact symbolic bounds when the reduced model is jointly affine,
        or result.solve('scip') for numerical bounds and restored witnesses.
        """
        return self.compile(limits).linearize(**options)
      
    def track_result_scip(self):
        """ 
        After running the optimization, this method obtains data from 
        the optimization routine and saves it into self.res_scip
        """
        res_lower = parse_whole_file_scip(self.scip_lower_filename)
        res_upper = parse_whole_file_scip(self.scip_upper_filename)
        if not res_lower and not res_upper:
            return None
        if not res_lower or not res_upper:
            return None
        import pandas as pd
        df_lower = pd.DataFrame(res_lower)
        df_upper = pd.DataFrame(res_upper)
        df_lower = df_lower.rename({'time':'seconds', 'dual': 'lb.dual',
                                    'primal':'lb.prim'}, axis = 1)
        df_upper = df_upper.rename({'time':'seconds', 'dual': 'ub.dual',
                                    'primal':'ub.prim'}, axis = 1)
        self.res_scip = df_lower.merge(df_upper, how = 'outer')
    
    def optimize_remove_numeric_lines(self):
        """Remove exact constant tautologies without rescaling contradictions."""
        from .model import Constraint, Polynomial
        constraints = []
        for row in self.constraints:
            if not row:
                continue
            relation = row[-1][0]
            if relation not in ('==', '<=', '>='):
                raise ValueError(f'Unsupported constraint relation: {relation}')
            try:
                expression = Polynomial.from_tokens(row[:-1], self.parameters)
            except (ValueError, ZeroDivisionError):
                # Legacy callers may declare parameter names after their rows.
                constraints.append(row)
                continue
            if not Constraint(expression, relation).is_tautology:
                constraints.append(row)
        self.constraints = constraints

    def optimize_add_param_value(self, parameter, value):
        """Fix a parameter, preserving its domain and defining equality."""
        from .polynomial import decimal_token
        if parameter not in self.parameters:
            raise ValueError(f"Unknown parameter: {parameter}")
        token = decimal_token(Fraction(str(value)))
        constraints = [change_constraint_parameter_value(row, parameter, token)
                       for row in self.constraints if row]
        constraints.append([[parameter], [decimal_token(-Fraction(token))], ['==']])
        self.constraints = constraints
        self._scip_bounds = None
        self.last_result = None

    def compile(self, limits=(None, None), *, eliminate_zeros=True):
        """Snapshot declarations and automatically remove provably zero variables.

        Set eliminate_zeros=False to inspect the unreduced legacy program.
        Original declarations remain available for recompilation/relaxation.
        """
        from dataclasses import replace
        from .model import MathematicalModel, Variable, Constraint, Polynomial, Term
        from .support import eliminate_zero_variables
        model = MathematicalModel.from_legacy(self.parameters, self.constraints, limits)
        eliminated = self._zero_variables
        if any(variable.name == 'objvar' for variable in eliminated):
            objective_variable = Variable('objvar', *limits)
            if ((objective_variable.lower is not None and objective_variable.lower > 0)
                    or (objective_variable.upper is not None and objective_variable.upper < 0)):
                # Later objective limits can conflict with a previously removed zero.
                model = replace(model, variables=model.variables + (objective_variable,),
                                constraints=model.constraints + (
                                    Constraint(Polynomial((Term(1, ('objvar',)),))),))
                eliminated = tuple(v for v in eliminated if v.name != 'objvar')
        model = replace(model, zero_variables=eliminated)
        return eliminate_zero_variables(model) if eliminate_zeros else model

    def run_scip(self, verbose=True, epsilon=-10, theta=0.01, maxtime=None,
                 debug=False, limits=(None, None), return_dgps=False, linearize=False):
        """Compile and solve through the SCIP backend; retain legacy return values."""
        from .backends.scip import ScipBackend
        self._scip_bounds = None
        self.last_result = None
        backend = ScipBackend()
        model = self.compile(limits).without_tautologies()
        self.last_reduction = model.linearize() if linearize else None
        if self.last_reduction is not None:
            model = self.last_reduction.model
        result = backend.solve(model,
                               verbose=verbose, epsilon=epsilon, theta=theta,
                               maxtime=maxtime, debug=debug, return_dgps=return_dgps)
        if self.last_reduction is not None:
            result = self.last_reduction.restore_result(result)
        self.last_result = result
        self._scip_backend = backend
        if debug:
            self.scip_lower_filename = backend.scip_lower_filename
            self.scip_upper_filename = backend.scip_upper_filename
        bounds = result.legacy_bounds()
        if all(endpoint.primal is not None and endpoint.dual is not None
               for endpoint in (result.lower, result.upper)):
            self._scip_bounds = tuple({key: endpoint[key] for key in ('dual', 'primal')}
                                      for endpoint in bounds[:2])
        return result.to_legacy()
    
    @property
    def M_lower(self):
        """Unsolved inspection model; native state is created only on access."""
        return self._scip_backend.M_lower

    @property
    def M_upper(self):
        """Unsolved inspection model; use last_result for worker solutions."""
        return self._scip_backend.M_upper

    def get_bounds_scip(self):
        """Return worker-reported bounds from the most recent successful run."""
        if self._scip_bounds is None:
            raise RuntimeError("Run run_scip successfully before requesting bounds")
        return tuple(dict(endpoint) for endpoint in self._scip_bounds)

    def run_couenne(self, verbose=True, filename=None, epsilon=0.01, theta=0.01):
        """Historical backend, retired because its results were not validated."""
        raise NotImplementedError("Couenne backend is retired; use run_scip()")

    def run_pyomo(self, solver_name='ipopt', verbose=True, parallel=False):
        """Historical backend, retired because its results were not validated."""
        raise NotImplementedError("Pyomo backend is retired; use run_scip()")

    def to_obj_pyomo(self):
        raise NotImplementedError("Pyomo export is unsupported; use to_pip()")

    def to_pip(self, filename, sense='max'):
        """Export the same compiled model consumed by the numerical backend."""
        from .exporters import write_pip
        write_pip(self.compile(), filename, sense)

    def to_cip(self):
        raise NotImplementedError("CIP export is unsupported; use to_pip()")

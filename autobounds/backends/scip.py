"""SCIP compilation, worker execution, monitoring and temporary-file ownership."""
import json
import os
import tempfile
from functools import reduce
from multiprocessing import get_context
from .scip_logs import solve_scip, parse_bounds_scip
from operator import eq, le, ge

get_symb_func = {'==': eq, '<=': le, '>=': ge}
from ..results import EndpointResult, SolverResult


class ScipBackend:
    """Numerical polynomial backend; symbolic parameters must first be bound."""
    def __init__(self):
        self.scip_lower_filename = None
        self.scip_upper_filename = None

    @staticmethod
    def supports(model):
        return not model.parameters

    @staticmethod
    def build_model(model, sense):
        from pyscipopt import Model, quicksum
        if model.parameters:
            raise ValueError("Bind symbolic parameters before SCIP compilation")
        compiled = Model()
        variables = {variable.name: compiled.addVar(
            variable.name,
            lb=None if variable.lower is None else float(variable.lower),
            ub=None if variable.upper is None else float(variable.upper),
        ) for variable in model.variables}

        def expression(polynomial):
            return quicksum(float(term.coefficient) * reduce(
                lambda a, b: a * b, (variables[name] for name in term.variables), 1)
                for term in polynomial.terms)

        for constraint in model.constraints:
            compiled.addCons(get_symb_func[constraint.relation](expression(constraint.expression), 0))
        # SCIP requires a linear objective; use a free auxiliary for general models.
        objective = expression(model.objective)
        if model.objective.degree > 1:
            name = '_autobounds_objective'
            while name in variables:
                name += '_'
            auxiliary = compiled.addVar(name, lb=None, ub=None)
            compiled.addCons(auxiliary == objective)
            objective = auxiliary
        compiled.setObjective(objective, sense=sense)
        return compiled

    @staticmethod
    def _result(bounds, dgps=None):
        return SolverResult(EndpointResult.from_dict(bounds[0]),
                            EndpointResult.from_dict(bounds[1]), bounds[2], bounds[3], dgps)

    def solve(self, model, *, verbose=True, epsilon=-10, theta=0.01,
              maxtime=None, debug=False, return_dgps=False):
        from contextlib import ExitStack
        if not self.supports(model):
            raise ValueError("SCIP requires a numeric model; bind symbolic parameters first")
        try:
            ctx = get_context('fork')
        except ValueError as exc:
            raise NotImplementedError(
                "run_scip requires multiprocessing fork; spawn is unsupported"
            ) from exc
        self.M_lower = self.build_model(model, 'minimize')
        self.M_upper = self.build_model(model, 'maximize')
        previous_paths = self.scip_lower_filename, self.scip_upper_filename
        processes = []
        with ExitStack() as stack:
            directory = '.' if debug else stack.enter_context(
                tempfile.TemporaryDirectory(prefix='autobounds-scip-'))
            self.scip_lower_filename = os.path.join(directory, '.lower.log')
            self.scip_upper_filename = os.path.join(directory, '.upper.log')
            dgp_paths = ([os.path.join(directory, '.lower.dgps.json'),
                          os.path.join(directory, '.upper.dgps.json')]
                         if return_dgps else [None, None])
            paths = [self.scip_lower_filename, self.scip_upper_filename]
            try:
                for path in paths + [path for path in dgp_paths if path is not None]:
                    with open(path, 'w'):
                        pass
                for compiled, path, dgp_path in zip(
                        (self.M_lower, self.M_upper), paths, dgp_paths):
                    process = ctx.Process(target=solve_scip, args=(compiled, path, False, dgp_path))
                    processes.append(process)
                    process.start()
                bounds = parse_bounds_scip(
                    *processes, filelower=paths[0], fileupper=paths[1],
                    epsilon=epsilon, theta=theta, maxtime=maxtime, verbose=verbose)
                dgps = None
                if return_dgps:
                    dgps = {name: self._load_dgp(path)
                            for name, path in zip(('lower', 'upper'), dgp_paths)}
                return self._result(bounds, dgps)
            except Exception as exc:
                raise RuntimeError(f"SCIP worker execution failed: {exc}") from exc
            finally:
                for process in processes:
                    try:
                        if process.is_alive():
                            process.terminate()
                        process.join()
                    except Exception:
                        # A process whose start failed cannot be joined.
                        pass
                if not debug:
                    self.scip_lower_filename, self.scip_upper_filename = previous_paths

    @staticmethod
    def _load_dgp(path):
        if path is None or not os.path.exists(path):
            return {"status": "missing", "values": {}}
        try:
            with open(path) as stream:
                data = json.load(stream)
            return data if isinstance(data, dict) else {"status": "invalid_format", "values": {}}
        except (OSError, ValueError):
            return {"status": "parse_error", "values": {}}

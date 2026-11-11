from fractions import Fraction
from functools import reduce
import io 
import random
import tempfile
from copy import deepcopy, copy
from multiprocessing import get_context
import time
import sys
from .ProgramUtils import *
import os
import json


alphanum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

class Program:
    """ This class
    will state a optimization program
    to be translated later to any 
    language of choice, pyscipopt (pip, pyscipopt(cip),
    pyomo, among others
    A program requires first parameters, 
    an objective function,
    and constraints
    Every method name starting with to_obj_ will solve the program directly in python.
    Method names starting with to_ will write the program to a file, which can 
    be read in particular solvers.
    """
    def __init__(self):
        self.parameters = [ ]
        self.constraints = [ tuple() ]
        self.res_scip = None
        self._scip_bounds = None
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

    def simplify_linear(self):
        """ Firstly, it divides constraints in linear and nonlinear
        Then, for each linear constraint, it tries to substitute them 
        insidee the nonlinear constraints, simplifying them """
        linear, nonlinear = [ ], [ ]
        for i in self.constraints:
            if is_linear(i):
                linear.append(i)
            else:
                nonlinear.append(i)
        linear, nonlinear = replace_linear(linear, nonlinear)
        self.constraints = linear + nonlinear
        

    def plot(self):
        self.track_result_scip()
        self.res_scip = self.res_scip.loc[lambda k: k.seconds != 'time']
        self.res_scip = self.res_scip.astype(float)
        self.res_scip = self.res_scip.sort_values(['seconds'])
        self.res_scip = self.res_scip.fillna(method = 'ffill')
        self.res_scip = self.res_scip.fillna(method = 'bfill')
        return plot_bounds(self.res_scip)
      
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
        df_lower = pd.DataFrame(res_lower)
        df_upper = pd.DataFrame(res_upper)
        df_lower = df_lower.rename({'time':'seconds', 'dual': 'lb.dual',
                                    'primal':'lb.prim'}, axis = 1)
        df_upper = df_upper.rename({'time':'seconds', 'dual': 'ub.dual',
                                    'primal':'ub.prim'}, axis = 1)
        self.res_scip = df_lower.merge(df_upper, how = 'outer')
    
    def optimize_remove_numeric_lines(self):
        """Remove true constant constraints, preserving exact contradictions.

        Decimal tokens are interpreted exactly when recognizing tautologies.
        False relations remain unchanged for the solver to evaluate with its
        numerical feasibility tolerance, including empirical roundoff residuals.
        Empty products represent one; an empty sum represents zero.
        """
        constraints = []
        for constraint in self.constraints:
            if not constraint:
                continue
            relation = constraint[-1][0]
            if relation not in ("==", ">=", "<="):
                raise ValueError(f"Unsupported constraint relation: {relation}")
            try:
                value = sum((reduce(lambda a, b: a * b,
                                    (Fraction(str(token)) for token in term),
                                    Fraction(1))
                             for term in constraint[:-1]), Fraction(0))
            except (ValueError, ZeroDivisionError):
                # At least one factor is a model parameter, not a constant.
                constraints.append(constraint)
                continue
            if not get_symb_func[relation](value, 0):
                constraints.append(constraint)
        self.constraints = constraints

    def optimize_add_param_value(self, parameter, value):
        """ 
        Replace directly one of the parameter by a certain value...
        That's ideal when we want to introduce a value directly
        """
        constraints2 = [ ]
        for i in self.constraints:
            constraints2.append(
                    change_constraint_parameter_value(i, parameter, value)
                    )
        self.constraints = constraints2
    
    def run_scip(self, verbose = True, epsilon = -10, theta = 0.01, maxtime = None, debug = False, limits = [None, None], return_dgps = False):
        """ We won't be using to_pip here,
        because we need the function to save into a .cip file


        debug = True won't remove files
        """
        from pyscipopt import Model, quicksum
        self._scip_bounds = None
        try:
            ctx = get_context('fork')
        except ValueError as exc:
            raise NotImplementedError(
                "run_scip requires multiprocessing fork; spawn is unsupported"
            ) from exc
        self.optimize_remove_numeric_lines()
        self.M_upper = Model()
        self.M_lower = Model() # Unfortunately we cannot use deepcopy with scip
        par_dict_upper = { }
        par_dict_lower = { }
        for p in self.parameters:
            if p != 'objvar':
                par_dict_lower[p] = self.M_lower.addVar(p, lb=0.0, ub=1.0)
                par_dict_upper[p] = self.M_upper.addVar(p, lb=0.0, ub=1.0)
            else:
                par_dict_upper[p] = self.M_upper.addVar(p, lb = limits[0], ub = limits[1])
                par_dict_lower[p] = self.M_lower.addVar(p, lb = limits[0], ub = limits[1])
        # Next loop is not elegant, needs refactoring
        for i, c in enumerate(self.constraints):
            self.M_upper.addCons(
                        get_symb_func[c[-1][0]](quicksum(mult_params_scip(k, par_dict_upper) for k in c[:-1]), 0)
                    )
            self.M_lower.addCons(
                        get_symb_func[c[-1][0]](quicksum(mult_params_scip(k, par_dict_lower) for k in c[:-1]), 0)
                    )
        self.M_upper.setObjective(par_dict_upper['objvar'], sense = 'maximize')
        self.M_lower.setObjective(par_dict_lower['objvar'], sense = 'minimize')
        old_lower, old_upper = self.scip_lower_filename, self.scip_upper_filename
        dgps_lower_file, dgps_upper_file = None, None
        if debug:
            self.scip_lower_filename = '.lower.log'
            self.scip_upper_filename = '.upper.log'
            open(self.scip_lower_filename, 'w').close()
            open(self.scip_upper_filename, 'w').close()
            if return_dgps:
                dgps_lower_file = '.lower.dgps.json'
                dgps_upper_file = '.upper.dgps.json'
                open(dgps_lower_file, 'w').close()
                open(dgps_upper_file, 'w').close()
        else:
            # Use temp files to avoid cwd side effects and cross-run collisions.
            lower_tmp = tempfile.NamedTemporaryFile(suffix='.log2n', delete=False)
            upper_tmp = tempfile.NamedTemporaryFile(suffix='.log2n', delete=False)
            lower_tmp.close()
            upper_tmp.close()
            self.scip_lower_filename = lower_tmp.name
            self.scip_upper_filename = upper_tmp.name
            if return_dgps:
                dgps_lower_tmp = tempfile.NamedTemporaryFile(suffix='.dgps.json', delete=False)
                dgps_upper_tmp = tempfile.NamedTemporaryFile(suffix='.dgps.json', delete=False)
                dgps_lower_tmp.close()
                dgps_upper_tmp.close()
                dgps_lower_file = dgps_lower_tmp.name
                dgps_upper_file = dgps_upper_tmp.name
        p_lower = ctx.Process(target=solve_scip, args=(self.M_lower, self.scip_lower_filename, False, dgps_lower_file))
        p_upper = ctx.Process(target=solve_scip, args=(self.M_upper, self.scip_upper_filename, False, dgps_upper_file))
        try:
            p_lower.start()
            p_upper.start()
            optim_data = parse_bounds_scip(p_lower, p_upper,
                            filelower = self.scip_lower_filename,
                            fileupper = self.scip_upper_filename, 
                                           epsilon = epsilon, theta = theta, maxtime = maxtime, verbose = verbose)
            if all("dual" in endpoint and "primal" in endpoint
                   for endpoint in optim_data[:2]):
                self._scip_bounds = tuple(
                    {key: endpoint[key] for key in ("dual", "primal")}
                    for endpoint in optim_data[:2]
                )
            if return_dgps:
                def _safe_load_dgps(path):
                    if path is None or not os.path.exists(path):
                        return {"status": "missing", "values": {}}
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                        return data if isinstance(data, dict) else {"status": "invalid_format", "values": {}}
                    except Exception:
                        return {"status": "parse_error", "values": {}}

                dgps = {"lower": {}, "upper": {}}
                dgps["lower"] = _safe_load_dgps(dgps_lower_file)
                dgps["upper"] = _safe_load_dgps(dgps_upper_file)
                return (optim_data, dgps)
            return optim_data
        except Exception as exc:
            raise RuntimeError(
                f"SCIP worker execution failed: {exc}"
            ) from exc
        finally:
            for proc in (p_lower, p_upper):
                try:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join()
                except Exception:
                    pass
            if not debug:
                for filename in (self.scip_lower_filename, self.scip_upper_filename):
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception:
                        pass
                for filename in (dgps_lower_file, dgps_upper_file):
                    try:
                        if filename is not None and os.path.exists(filename):
                            os.remove(filename)
                    except Exception:
                        pass
                self.scip_lower_filename, self.scip_upper_filename = old_lower, old_upper
    
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

    def to_pip(self, filename, sense = 'max'):
        if isinstance(filename, str):
            filep = open(filename, 'w')
        elif isinstance(filename, io.StringIO):
            filep = filename
        else:
            raise Exception('Filename type not accepted!')
        sense = 'MAXIMIZE' if sense == 'max' else 'MINIMIZE'
        filep.write(sense + '\n' + '  obj: objvar' + '\n')
        filep.write('\nSUBJECT TO\n')
        for i, c in enumerate(self.constraints):
            if len(c) == 0:
                continue
            filep.write('a' + str(i) + ': ' + ' + '.join([pip_join_expr(k, self.parameters) 
                for k in c[:-1] ]) + ' ' + fix_symbol_pip(c[-1][0]) + ' 0\n')
        filep.write('\nBOUNDS\n')
        for p in self.parameters:
            if p != 'objvar':
                filep.write(f'  0 <= {p} <= 1\n')
            else:
                filep.write(f'  -1 <= {p} <= 1\n')
        filep.write('\nEND')
        filep.close()
    
    def to_cip(self):
        raise NotImplementedError("CIP export is unsupported; use to_pip()")

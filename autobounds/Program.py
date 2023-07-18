from functools import reduce
import io 
from copy import deepcopy, copy
from multiprocessing import Process,Pool,Manager
import time
import sys
from .ProgramUtils import *

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
    
    def optimize_remove_numeric_lines(self):
        """ 
        All lines of the type [[0.25], [-0.25], [==]. []] 
        i.e. no numeric parameter is included ,
        should be removed
        """
        constraints2 = [ ]
        for i in self.constraints:
            if not all([ test_string_numeric_list(j) for j in i ]):
                constraints2.append(i)
        self.constraints = constraints2
    
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
    
    def run_scip(self, verbose = True, filename = None, epsilon = 0.01, theta = 0.01, maxtime = None):
        """ We won't be using to_pip here,
        because we need the function to save into a .cip file
        """
        from pyscipopt import Model
        self.M_upper = Model()
        self.M_lower = Model() # Unfortunately we cannot use deepcopy with scip
        par_dict_upper = { }
        par_dict_lower = { }
        for p in self.parameters:
            if p != 'objvar':
                par_dict_lower[p] = self.M_lower.addVar(p, lb=0.0, ub=1.0)
                par_dict_upper[p] = self.M_upper.addVar(p, lb=0.0, ub=1.0)
            else:
                par_dict_upper[p] = self.M_upper.addVar(p, lb = None, ub = None)
                par_dict_lower[p] = self.M_lower.addVar(p, lb = None, ub = None)
        # Next loop is not elegant, needs refactoring
        for i, c in enumerate(self.constraints):
            self.M_upper.addCons(
                        get_symb_func[c[-1][0]](sum([ mult_params_scip(k, par_dict_upper) for k in c[:-1] ]), 0)
                    )
            self.M_lower.addCons(
                        get_symb_func[c[-1][0]](sum([ mult_params_scip(k, par_dict_lower) for k in c[:-1] ]), 0)
                    )
        self.M_upper.setObjective(par_dict_upper['objvar'], sense = 'maximize')
        self.M_lower.setObjective(par_dict_lower['objvar'], sense = 'minimize')
        p_lower = Process(target=lambda k: solve_scip(k, 'lower'), args=[self.M_lower])
        p_upper = Process(target=lambda k: solve_scip(k, 'upper'), args=[self.M_upper])
        p_lower.start()
        p_upper.start()
        optim_data = parse_bounds_scip(p_lower, p_upper, epsilon = epsilon, theta = theta, maxtime = maxtime)
        return optim_data
    
    def get_bounds_scip(self):
        return (
                {'dual': self.M_lower.getDualbound(), 'primal': self.M_lower.getPrimalbound() },
                {'dual': self.M_upper.getDualbound(), 'primal': self.M_upper.getPrimalbound() }
                )
    
    def run_couenne(self, verbose = True, filename = None, epsilon = 0.01, theta = 0.01):
        """ This method runs programs directly in python using pyomo and couenne
        """
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
        M = pyo.ConcreteModel()
        solver = pyo.SolverFactory('couenne')
        for p in self.parameters:
            if p != 'objvar':
                setattr(M, p, pyo.Var(bounds = (0,1)))
            else:
                setattr(M, p, pyo.Var())
        # Next loop is not elegant, needs refactoring
        for i, c in enumerate(self.constraints):
            setattr(M, 'c' + str(i), 
                    pyo.Constraint(expr = 
                        get_symb_func[c[-1][0]](sum([ mult_params(self.parameters, k, M ) for k in c[:-1] ]), 0)
                    )
            )
        self.M_upper = deepcopy(M)
        self.M_lower = deepcopy(M)
        self.M_upper.obj = pyo.Objective(expr = self.M_upper.objvar, sense = pyo.maximize)
        self.M_lower.obj = pyo.Objective(expr = self.M_lower.objvar, sense = pyo.minimize)
        open('.lower.log','w').close()
        open('.upper.log','w').close()
        p_lower = Process(target=solve1, args=(solver, self.M_lower,'lower', verbose)) 
        p_upper = Process(target=solve1, args=(solver, self.M_upper,'upper', verbose)) 
        p_lower.start()
        p_upper.start()
        optim_data = parse_bounds(p_lower, p_upper, filename, epsilon = epsilon, theta = theta)
        return optim_data
   
    def run_pyomo(self, solver_name = 'ipopt', verbose = True, parallel = False):
        """ This method runs program directly in python using pyomo
        """
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
        M = pyo.ConcreteModel()
        solver = pyo.SolverFactory(solver_name)
        solve = lambda a: solver.solve(a, tee = verbose)
        for p in self.parameters:
            if p != 'objvar':
                setattr(M, p, pyo.Var(bounds = (0,1)))
            else:
                setattr(M, p, pyo.Var(bounds = (-1, 1)))
        # Next loop is not elegant, needs refactoring
        for i, c in enumerate(self.constraints):
            setattr(M, 'c' + str(i), 
                    pyo.Constraint(expr = 
                        get_symb_func[c[-1][0]](sum([ mult_params(self.parameters, k, M ) for k in c[:-1] ]), 0)
                    )
            )
        self.M1 = deepcopy(M)
        self.M2 = deepcopy(M)
        self.M1.obj = pyo.Objective(expr = self.M1.objvar, sense = pyo.maximize)
        self.M2.obj = pyo.Objective(expr = self.M2.objvar, sense = pyo.minimize)
        if parallel:
            with Pool(None, initializer=worker_init, initargs=(solve,)) as p:
                p.map(worker, [self.M1,self.M2])
        else:
            results = list(map(solve, [self.M1,self.M2]))
        solver.solve(self.M1, tee = verbose)
        solver.solve(self.M2, tee = verbose)
        lower_bound = pyo.value(self.M2.objvar)
        upper_bound = pyo.value(self.M1.objvar)
        return (lower_bound, upper_bound)
   
    def to_obj_pyomo(self):
        pass
    
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
        pass


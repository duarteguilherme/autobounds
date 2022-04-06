from functools import reduce
import io 
from copy import deepcopy


pyomo_symb = {
        '==': lambda a,b: a== b,
        '<=': lambda a,b: a<= b,
        '>=': lambda a,b: a>= b,
        '<': lambda a,b: a < b,
        '>': lambda a,b: a > b,
        }

fix_symbol_pip = lambda a: '=' if a == '==' else a

def pip_join_expr(expr, params):
    """ 
    It gets an expr and if there is a coefficient, it 
    separates without using * .
    It is required as a simple list join is insufficient 
    to put program in pip format
    """
    coef = ''.join([x for x in expr if x not in params ])
    expr_rest = ' * '.join([ x for x in expr if x in params ])
    coef = coef + ' ' if coef != '' and expr_rest != '' else coef 
    return coef + expr_rest

def test_pip_join_expr():
    assert pip_join_expr(['0.5', 'X00.Y00'], ['X00.Y00', 'Z1', 'Z0']) == '0.5 X00.Y00'
    assert pip_join_expr(['0.5'], ['X00.Y00', 'Z1', 'Z0']) == '0.5'
    assert pip_join_expr(['X00.Y00'], ['X00.Y00', 'Z1', 'Z0']) == 'X00.Y00'
    assert pip_join_expr(['0.5', 'X00.Y00', 'Z1'], ['X00.Y00', 'Z1', 'Z0']) == '0.5 X00.Y00 * Z1'


def mult_params_pyomo(params, k, M):
    """ Function to be used in run_pyomo
    Get parameters and multiply them
    """
    return reduce(lambda a, b: a * b, 
    [ getattr(M, r) if r in params else float(r)  
        for r in k ])

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
    
    def run_pyomo(self, solver_name = 'ipopt', verbose = True):
        """ This method runs program directly in python using pyomo
        """
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
        M = pyo.ConcreteModel()
        solver = pyo.SolverFactory(solver_name)
        for p in self.parameters:
            if p != 'objvar':
                setattr(M, p, pyo.Var(bounds = (0,1)))
            else:
                setattr(M, p, pyo.Var())
        # Next loop is not elegant, needs refactoring
        for i, c in enumerate(self.constraints):
            setattr(M, 'c' + str(i), 
                    pyo.Constraint(expr = 
                        pyomo_symb[c[-1][0]](sum([ mult_params_pyomo(self.parameters, k, M ) for k in c[:-1] ]), 0)
                    )
            )
        M1 = deepcopy(M)
        M2 = deepcopy(M)
        M1.obj = pyo.Objective(expr = M1.objvar, sense = pyo.maximize)
        M2.obj = pyo.Objective(expr = M2.objvar, sense = pyo.minimize)
        solver.solve(M1, tee = verbose)
        solver.solve(M2, tee = verbose)
        lower_bound = pyo.value(M2.objvar)
        upper_bound = pyo.value(M1.objvar)
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


from functools import reduce
import io 
from copy import deepcopy
from multiprocessing import Process,Pool
import time
import sys

pyomo_symb = {
        '==': lambda a,b: a== b,
        '<=': lambda a,b: a<= b,
        '>=': lambda a,b: a>= b,
        '<': lambda a,b: a < b,
        '>': lambda a,b: a > b,
        }

fix_symbol_pip = lambda a: '=' if a == '==' else a


# Workaround in order to use lambda inside multiprocessing
_func = None

def worker_init(func):
  global _func
  _func = func
  

def worker(x):
  return _func(x)

def solve1(solver, model, sensetype, verbose):
    sys.stdout = open('.' + sensetype + '.log', 'w', buffering = 1)
    solver.solve(model, tee = verbose)
 

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



def parse_cbc_line(line, sign = 1):
    """ 
    Parses particular rows in parse_particular_bound
    It returns dual, primal, and time.
    It only works for cbc. Not for cbl

    Due to a particularity of couenne, if one intends to get upper bound, 
    sign must be -1
    """
    result_data = {
            'primal': sign*float(line.split('on tree,')[1].split('best solution')[0].strip()),
            'dual': sign*float(line.split('best possible')[1].split('(')[0].strip()),
            'time': float(line.split('(')[-1].split('seconds')[0].strip())
            }
    return result_data
    

def test_string_numeric(string):
    """
    Test if a string is numeric, or equal to "==", ">=", or "<="
    """
    if ( string == '==' ) or ( string == '>=' ) or (string == '<=' ):
        return True
    try:
        float(string)
        return True
    except:
        return False


def parse_particular_bound(filename, n_bound):
    """ Read any of ".lower.log" or
    ".upper.log" and it returns 
    data
    """
    sign = 1 if filename == ".lower.log" else -1
    with open(filename) as f:
        data = f.readlines()
    datarows = [ x 
            for x in data if x.strip().startswith('Cbc') and 'After' in x ]
    datarows = [ parse_cbc_line(x, sign) 
            for x in datarows ]
    if len(datarows) > n_bound:
        return (len(datarows), datarows[(n_bound-1):])
    else:
        return (n_bound, {})


def get_final_bound(filename):
    with open(filename,'r') as f: 
        data = f.readlines()
    sign = 1 if filename == ".lower.log" else -1
    result = {}
    result['primal'] = sign*float([ k for k in data if k.startswith('Upper bound:')][-1]
            .split(':')[1].split('(')[0].strip())
    result['dual'] = sign*float([ k for k in data if k.startswith('Lower bound:')][-1]
            .split(':')[1].split('(')[0].strip())
    result['time'] = float([ k for k in data if k.startswith('Total solve time')][-1]
            .split(':')[1].split('s')[0].strip())
    return result



def check_process_end(p, filename):
    with open(filename,'r') as f: 
        data = f.readlines()
    if any([x for x in data if '"Finished"' in x ]):
        print("Problem is finished! Returning final values")
        p.terminate()
        return 1
    if any([x for x in data if 'Problem infeasible' in x ]):
        print("Problem is infeasible. Returning without solutions")
        p.terminate()
        return 0
    else:
        return -1

def change_constraint_parameter_value(constraint, parameter, value):
    constraint2 = [ j.copy() for j in constraint ]
    const = [ ]
    for i in constraint2:
        if parameter in i:
            if test_string_numeric(i[0]):
                i[0] =  str(float(i[0]) * value) 
            else:
                i = [  str(value)  ] + i
        const.append([ k for k in i if k != parameter ])
    return const



def parse_bounds(p_lower, p_upper, filename = None, epsilon = 0.01, theta = 0.01):
    """ 
    Read files ".lower.log" and ".upper.log" each 1000 miliseconds 
    and retrieve data on dual and primal bounds.
    Also, it returns sucessful or not
    - Input: two multiprocessing processes, and thresholds for epsilon and theta
    - Output: lower bound dict, upper bound dict, current_theta, current_epsilon or 
    ( {}, {}, -1, -1 ) if it fails.
    - States:
        * n_upper = number of rows in upper data
        * n_lower = number of rows in lower data
    """
    time.sleep(0.5)
    total_lower,total_upper = [], []
    n_lower, n_upper = 0,0
    current_theta, current_epsilon = 9999, 9999
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(f"bound,primal,dual,time\n")
    while True:
        n_lower, partial_lower = parse_particular_bound('.lower.log', n_lower)
        n_upper, partial_upper = parse_particular_bound('.upper.log', n_upper)
        total_lower += partial_lower
        total_upper += partial_upper
        if len(partial_lower) > 0:
            for i in partial_lower:
                print(f"LOWER BOUND: # -- Primal: {i['primal']} / Dual: {i['dual']} / Time: {i['time']} ##")
                if filename is not None:
                    with open(filename, 'a') as f:
                        f.write(f"lb,{i['primal']},{i['dual']},{i['time']}\n")
        if len(partial_upper) > 0:
            for j in partial_upper:
                print(f"UPPER BOUND: # -- Primal: {j['primal']} / Dual: {j['dual']} / Time: {j['time']} ##")
                if filename is not None:
                    with open(filename, 'a') as f:
                        f.write(f"ub,{j['primal']},{j['dual']},{j['time']}\n")
        end_lower = check_process_end(p_lower, '.lower.log')
        end_upper = check_process_end(p_upper, '.upper.log')
        if len(total_lower) > 0 and len(total_upper) > 0:
            current_theta = total_upper[-1]['dual'] - total_lower[-1]['dual']
            gamma = abs(total_upper[-1]['primal'] - total_lower[-1]['primal']) 
            current_epsilon = current_theta/gamma - 1 if gamma != 0 else 99999999
            print(f"CURRENT THRESHOLDS: # -- Theta: {current_theta} / Epsilon: {current_epsilon} ##")
            if current_theta <  theta or current_epsilon < epsilon:
                p_lower.terminate()
                p_upper.terminate()
                break
        if end_lower != -1 and end_upper != -1:
            break
        time.sleep(0.1)
    # Checking bounds if problem is finished
    if end_lower == 1 or end_upper == 1: 
        if end_lower == 1:
            i = get_final_bound('.lower.log')
        if end_upper == 1:
            j = get_final_bound('.upper.log')
#        i,j = get_final_bound('.lower.log'), get_final_bound('.upper.log')
        current_theta = j['dual'] - i['dual']
        current_epsilon = current_theta/abs(j['primal'] - i['primal']) - 1
    else:
        if end_lower == 0 and end_upper == 0:
            i, j, current_theta, current_epsilon = {}, {},-1,-1
    i['end'] = end_lower
    j['end'] = end_upper
    if filename is not None:
        with open(filename, 'a') as f:
            f.write(f"lb,{i['primal']},{i['dual']},{i['time']}\n")
            f.write(f"ub,{j['primal']},{j['dual']},{j['time']}\n")
    return (i, j, current_theta, current_epsilon)
    

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
            if not all([ test_string_numeric(j) for j in i ]):
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
                        pyomo_symb[c[-1][0]](sum([ mult_params_pyomo(self.parameters, k, M ) for k in c[:-1] ]), 0)
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
                        pyomo_symb[c[-1][0]](sum([ mult_params_pyomo(self.parameters, k, M ) for k in c[:-1] ]), 0)
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


from functools import reduce
import io 
from copy import deepcopy, copy
from multiprocessing import Process,Pool,Manager
import time
import sys

get_symb_func = {
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
    """ To be used with couenne """
    sys.stdout = open('.' + sensetype + '.log', 'w', buffering = 1)
    solver.solve(model, tee = verbose)

def solve_scip(model, sensetype, verbose = False):
    """ To be used with scip """
    model.redirectOutput()
    sys.stdout = open('.' + sensetype + '.log', 'w', buffering = 1)
    model.optimize()
    

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


def mult_params(params, k, M):
    """ Function to be used in run_pyomo
    Get parameters and multiply them
    """
    return reduce(lambda a, b: a * b, 
    [ getattr(M, r) if r in params else float(r)  
        for r in k ])

def mult_params_scip(k, par_dict):
    """ Function to be used in run_pyomo
    Get parameters and multiply them
    """
    return reduce(lambda a, b: a * b, 
    [ par_dict[r] if r in par_dict.keys() else float(r)  
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

def test_string_numeric_list(lst):
    """ Same as test_string_numeric, but with lists
    """
    if len(lst) == 1:
        return test_string_numeric(lst[0])
    elif len(lst) == 0:
        return True
    else:
        return False

def tofloat(num):
    try:
        num = float(num)
    except:
        pass
    num = -10 if num == '--' else num # Fixing problem with scip parser
    return num

def parse_particular_bound_scip(filename, n_bound):
    """ Read any of ".lower.log" or
    ".upper.log" and it returns 
    data
    """
    with open(filename) as f:
        data = f.readlines()
    datarows = [ x 
            for x in data if len(x.split('|')) > 5 ]
    if len(datarows) > n_bound:
        res = datarows[-1].split('|')
        res = res[0:1] + res[-4:-2] 
        return (len(datarows), [{  
                                'time': tofloat(res[0].strip().split('s')[0]),
                                'dual': tofloat(res[1].strip()), 
                                'primal': tofloat(res[2].strip()) }])
    else:
        return (n_bound, {})


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


def get_final_bound_scip(filename):
    with open(filename,'r') as f: 
        data = f.readlines()
    result = {}
    result['primal'] = float([ k for k in data if k.startswith('Primal Bound')][-1]
            .split(':')[1].split('(')[0].strip())
    result['dual'] = float([ k for k in data if k.startswith('Dual Bound')][-1]
            .split(':')[1].split('(')[0].strip())
    result['time'] = float([ k for k in data if k.startswith('Solving Time')][-1]
            .split(':')[1].split('s')[0].strip())
    return result

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


def check_process_end_scip(p, filename):
    with open(filename,'r') as f: 
        data = f.readlines()
    if any([x for x in data if "[optimal solution found]" in x ]):
        print("Problem is finished! Returning final values")
        p.terminate()
        return 1
    if any([x for x in data if 'problem is solved [infeasible]' in x ]):
        print("Problem is infeasible. Returning without solutions")
        p.terminate()
        return 0
    else:
        return -1

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


def parse_bounds_scip(p_lower, p_upper, filename = None, epsilon = 0.01, theta = 0.01, maxtime = None):
    time.sleep(0.5)
    init_time = time.time()
    total_lower, total_upper = [ ], []
    n_lower, n_upper = 0, 0
    current_theta, current_epsilon = 9999, 9999
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(f"bound,primal,dual,time\n")
    while True:
        n_lower, partial_lower = parse_particular_bound_scip('.lower.log', n_lower)
        n_upper, partial_upper = parse_particular_bound_scip('.upper.log', n_upper)
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
        end_lower = check_process_end_scip(p_lower, '.lower.log')
        end_upper = check_process_end_scip(p_upper, '.upper.log')
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
        if maxtime is not None:
            if time.time() - init_time > maxtime:
                break
        time.sleep(1)
    # Checking bounds if problem is finished
    if end_lower == 1 or end_upper == 1: 
        if end_lower == 1:
            i = get_final_bound_scip('.lower.log')
        if end_upper == 1:
            j = get_final_bound_scip('.upper.log')
        current_theta = j['dual'] - i['dual']
        gamma = abs(j['primal'] - i['primal'])
        current_epsilon = current_theta/gamma - 1 if gamma != 0 else 99999999
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

def parse_bounds(p_lower, p_upper, filename = None, epsilon = 0.01, theta = 0.01):
    """ 
    For couenne
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
        time.sleep(1)
    # Checking bounds if problem is finished
    if end_lower == 1 or end_upper == 1: 
        if end_lower == 1:
            i = get_final_bound('.lower.log')
        if end_upper == 1:
            j = get_final_bound('.upper.log')
#        i,j = get_final_bound('.lower.log'), get_final_bound('.upper.log')
        current_theta = j['dual'] - i['dual']
        gamma = abs(j['primal'] - i['primal']) 
        current_epsilon = current_theta/gamma - 1 if gamma != 0 else 99999999
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
    

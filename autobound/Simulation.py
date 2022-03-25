import pandas as pd 
from autobound.DAG import DAG
from autobound.SCM import SCM
from autobound.bounds import causalProgram
from pyscipopt import quicksum
from autobound.utils import *

N = 10000 # Size of simulations 


def unconfounded_simple_model():
    """ it must return a dag and an estimand -- 
    for instance the ate
    Estimand exprs like 
    { 'sign': -1,
    'var': ('Y', 1),
    'do': ('X', 0) }
    (a,b,c) -> a is the sign of the expr, b is the variable,
    and c is the value of this variable
    """
    dag = DAG()
    dag.from_structure("X -> Y")
    return (dag, prepare_ate('Y', 'X'))

def confounded_simple_model():
    dag = DAG()
    dag.from_structure("X -> Y, U -> X, U -> Y", unob = "U")
    return (dag, prepare_ate('Y', 'X'))

def balke_pearl():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    return (dag, prepare_ate('Y', 'X'))

def front_door():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U1 -> Z, U1 -> Y", unob = "U1")
    return (dag, prepare_ate('Y', 'Z'))

def napkin():
    dag = DAG()
    dag.from_structure("""W -> Z, Z -> X, X -> Y, Uxw -> X, U -> W, 
            Uxw -> W,
            U -> Y""", 
        unob = "U,Uxw")
    dag.find_c_components()
    return (dag, prepare_ate('Y', 'X'))

def selection_graph():
    dag = DAG()
    dag.from_structure("X -> Y, U -> X, U -> Y, Y -> S",
            unob = "U")
    return (dag, prepare_ate('Y', 'X'))


def get_bound(dag, m, estimand, typeb = 'minimize'):
    """ typeb must indicate the type of the bound.
    There are two types: 'minimize' for lower bound
    and 'maximize' for upper bound
    """
    program = causalProgram(typeb)
    program.from_dag(dag)
    program.add_prob_constraints()
    program.add_indep_constraints()
    program.program.setRealParam('limits/gap', 0.5)
    introduce_prob_into_progr(program,
    get_probability_from_model(m))
    program.set_obj(parse_estimand(program, estimand))
    program.program.writeProblem('/home/beta/check.cip')
    program.program.optimize()
    sol = program.program.getBestSol()
    sol = program.program.getSolObjVal(sol)
    return sol



def test_model(func):
    dag, estimand = func()
    m = simulate_model(dag)
    get_probability_from_model(m, overlap = True)
#    input("Continue?:")
    lb = get_bound(dag, m, estimand, 'minimize')
    ub = get_bound(dag, m, estimand, 'maximize')
    estimand_value  = get_c_estimand_value(m, estimand)
    return {'lb':lb,  'estimand': estimand_value, 'ub': ub}


test_model(confounded_simple_model)
test_model(unconfounded_simple_model)
test_model(balke_pearl)
test_model(front_door)
test_model(napkin)




def get_bound_from_csv(dag, filename, estimand, typeb = 'minimize'):
    """ typeb must indicate the type of the bound.
    There are two types: 'minimize' for lower bound
    and 'maximize' for upper bound
    """
    program = causalProgram(typeb)
    program.from_dag(dag)
    program.add_prob_constraints()
    p_table = pd.read_csv(filename)
    introduce_prob_into_progr(program,p_table)
    program.set_obj(parse_estimand(program, estimand))
    program.program.optimize()
    sol = program.program.getBestSol()
    sol = program.program.getSolObjVal(sol)
    program.program.writeProblem('/home/beta/check.cip')
    return sol



def test_from_file(func, filename):
    dag, estimand = func()
    lb = get_bound_from_csv(dag, filename, estimand, 'minimize')
    ub = get_bound_from_csv(dag, filename, estimand, 'maximize')
    return {'lb':lb,  'ub': ub}


filename = "selection_obsqty.csv"
test_from_file(selection_graph, filename)


import pandas as pd 
from autobound.DAG import DAG
from autobound.SCM import SCM
from autobound.bounds import causalProgram
from pyscipopt import quicksum

N = 10000 # Size of simulations 

def prepare_ate(var, do):
    """ Prepare ATE expr for simulations
    """
    expr1  = { 'sign': 1,'var': (var, 1),'do': (do, 1) }
    expr2  = { 'sign': -1,'var': (var, 1),'do': (do, 0) }
    return (expr1, expr2)

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
    dag.from_structure("""W -> Z, Z -> X, X -> Y, U -> X, U -> W, 
            U -> Y""", 
            unob = "U")
    return (dag, prepare_ate('Y', 'X'))

def selection_graph():
    dag = DAG()
    dag.from_structure("X -> Y, U -> X, U -> Y, Y -> S",
            unob = "U")
    return (dag, prepare_ate('Y', 'X'))

def get_probability_from_model(m, intervention = {}, overlap = False):
    data = m.draw_sample(intervention = intervention)
    columns_to_group = [x for x in data.columns  if x not in m.u_data.keys() ]
    data = (data
            .drop(m.u_data.keys(), axis = 1)
            .assign(P = 1)
            .groupby(columns_to_group)
            .agg('count')
            .assign(P = lambda v: v['P'] / N)
            .reset_index()
            )
    if overlap == True and len(data) < (2**len(m.V)):
        print("*" * 30)
        print("Be Careful! It violates overlap")
        print("*" * 30)
        print(data)
    return data

def get_estimand_value(m, e):
    prob_table = get_probability_from_model(m, 
                intervention = dict([e['do']]))
    filter_v = dict([e['var']])
    filtered = (prob_table
            .loc[(prob_table[list(filter_v)] == pd.Series(filter_v)).all(axis=1)])
    if len(filtered) > 1:
        vars = [e['var'][0]] + [e['do'][0]]
        filtered = filtered.groupby(vars).sum()
    if len(filtered) == 0:
        return 0
    return e['sign']*filtered.P.iloc[0]

def get_c_estimand_value(m, estimand):
    value = 0
    for e in estimand:
        value += get_estimand_value(m, e)
    return value

def simulate_model(dag):
    model = SCM()
    model.from_dag(dag)
    model.sample_u(N)
    return model


def parse_estimand(program, estimand):
    return quicksum(
            [ e['sign'] * quicksum(
                program.get_expr(**{ key: '='.join([ str(part) for part in e[key] ])
                            for key in ['var','do'] }) )
                                for e in estimand ])


def introduce_prob_into_progr(program, prob_table):
    prob_data = prob_table.T.to_dict().values()
    for p in prob_data:
        program.program.addCons(
            quicksum(program.get_expr(var = 
                    ','.join(
                        [ tupl[0] + '=' + str(int(tupl[1])) for tupl in list(p.items())[:-1] ]))
                ) ==
            list(p.items())[-1][1]
        )


def get_bound(dag, m, estimand, typeb = 'minimize'):
    """ typeb must indicate the type of the bound.
    There are two types: 'minimize' for lower bound
    and 'maximize' for upper bound
    """
    program = causalProgram(typeb)
    program.from_dag(dag)
    program.add_prob_constraints()
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
    lb = get_bound(dag, m, estimand, 'minimize')
    ub = get_bound(dag, m, estimand, 'maximize')
    estimand_value  = get_c_estimand_value(m, estimand)
    get_probability_from_model(m, overlap = True)
    return {'lb':lb,  'estimand': estimand_value, 'ub': ub}


test_model(confounded_simple_model)
test_model(unconfounded_simple_model)
test_model(balke_pearl)
test_model(front_door)
test_model(napkin)
#program.get_expr("Y = 1", "X = 1")
#program.get_expr("Y = 1, X = 1")
#program.get_expr("Y = 1")

def get_bound(dag, m, estimand, typeb = 'minimize'):
    """ typeb must indicate the type of the bound.
    There are two types: 'minimize' for lower bound
    and 'maximize' for upper bound
    """
    program = causalProgram(typeb)
    program.from_dag(dag)
    program.add_prob_constraints()
    introduce_prob_into_progr(program,
    get_probability_from_model(m))
    program.set_obj(parse_estimand(program, estimand))
    program.program.optimize()
    sol = program.program.getBestSol()
    sol = program.program.getSolObjVal(sol)
    program.program.writeProblem('/home/beta/check.cip')
    return sol



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

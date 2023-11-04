from autobounds.autobounds.causalProblem import causalProblem
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Program import change_constraint_parameter_value
from autobounds.autobounds.ProgramUtils import is_linear
from autobounds.autobounds.Query import Query
import io
import time
import pandas as pd


def test_is_linear():
    df = pd.DataFrame(
            {'Z': [0,0,0,0,1,1,1,1],
             'W': [0,0,1,1,0,0,1,1],
             'Y': [0,1,0,1,0,1,0,1],
          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
             })
    dag = DAG()
    dag.from_structure('U -> Z, Z -> W, U -> Y, W -> Y')
    pro = causalProblem(dag)
    pro.load_data(df)
    pro.set_ate('W','Y')
    pro.add_prob_constraints()
    program = pro.write_program()
    assert is_linear(program.constraints[-1])

def test_separation_lin_nonlin():
    df = pd.DataFrame(
            {'Z': [0,0,0,0,1,1,1,1],
             'W': [0,0,1,1,0,0,1,1],
             'Y': [0,1,0,1,0,1,0,1],
          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
             })
    dag = DAG()
    dag.from_structure('U -> Z, Z -> W, U -> Y, W -> Y')
    pro = causalProblem(dag)
    pro.add_constraint(pro.query('Z(U=0)=0') - Query(0.43))
    pro.add_constraint(pro.query('Z(U=0)=1') - Query(0.25))
    pro.add_constraint(pro.query('Z(U=1)=0') - Query(0.35))
    pro.add_constraint(pro.query('Z(U=1)=1') - Query(0.15))
    pro.load_data(df)
    pro.set_ate('W','Y')
    pro.add_prob_constraints()
    program = pro.write_program()
    constraints1 = program.constraints
    program.simplify_linear()
    constraints2 = program.constraints
    print(len(constraints1))
#    assert is_linear(program.constraints[-1])




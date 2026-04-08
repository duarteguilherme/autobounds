from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.closedProblem import (
     closedProblem, get_cover_addition, is_inconclusive, get_cover_subtraction,
     Gset
)
from autobounds.autobounds.Parser import Parser
import pandas as pd
import io
from copy import deepcopy
from collections import Counter



dag = DAG()
dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
problem = closedProblem(dag)





def test_iv4():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag, {'Z': 4})
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Y,X', 'Z=2')
    problem.load_data('Y,X', 'Z=3')
    problem.load_data('Z')
    problem.set_estimand(problem.query('Y(X=1)=1'))
    problem.find_solutions()




def test_sensitivity():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, Z -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Z')
#    problem.load_data('Y,X,Z')
    problem.set_estimand(problem.query('Y(X=1)=1&Z(X=1)=1'))
    problem.add_sens_parameter('mono', problem.query('X(Z=1)=0&X(Z=0)=1'))
    print(problem.data)
    problem.find_solutions()
#    problem.find_bounds_subtraction(problem.query('Y(X=0)=1'))
#    problem.read_solution()


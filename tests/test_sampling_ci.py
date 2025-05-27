from autobounds.autobounds.causalProblem import causalProblem, respect_to
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Q import Q
import pandas as pd
import numpy as np

def test_program_read_data_no_covariates():
    pass

def test_program_read_data_covariates():
    dag = DAG("D -> Y")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') + p('Y(D=0)=1', -1))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
    df = pd.DataFrame({
        'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
        'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
        'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
    })
    problem.read_data(df, covariates = ['X'])
 #   z = problem.write_program()
 #   res = z.run_scip(maxtime = 5)

def test_program_calculate_ci():
    dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') + p('Y(D=0)=1', -1))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
    df = pd.DataFrame({
        'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
        'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
        'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
    })
    problem.read_data(df, covariates = ['X'])
    np.random.seed(19103)
    res = problem.calculate_ci(ncoef = 2)
    assert res[0] > -0.55 and res[0] < -0.54
    assert res[1] < 0.001


def test_program_solve():
    dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') + p('Y(D=0)=1', -1))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
    df = pd.DataFrame({
        'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
        'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
        'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
    })
    problem.read_data(df, covariates = ['X'])
    print(problem.solve())
    # np.random.seed(19103)
    # res = problem.calculate_ci(ncoef = 2)
    # assert res[0] > -0.73 and res[0] < -0.72
    # assert res[1] < 0.001

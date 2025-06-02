from autobounds.autobounds.causalProblem import causalProblem, respect_to
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Q import Q
import pandas as pd
import numpy as np


dag = DAG("D -> Y")
df = pd.DataFrame({
    'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
    'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
    'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
})

def test_program_read_data_no_covariates():
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
        read_data(df, covariates = ['X'])
    res = problem.calculate_ci(categorical = True)


def test_program_read_data_covariates():
    dag = DAG("D -> Y")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
    df = pd.DataFrame({
        'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
        'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
        'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
    })
    problem.read_data(df, covariates = ['X'])


def test_program_calculate_ci():
    dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
    df = pd.DataFrame({
        'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
        'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
        'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
    })
    problem.read_data(df, covariates = ['X'])
    np.random.seed(19103)
    res0 = problem.calculate_ci(categorical = False, nsamples = 2)
    res = ( np.quantile(res0[0], 0.025), np.quantile(res0[1], 0.975) )
    assert res[0] > -0.55 and res[0] < -0.54
    assert res[1] < 0.001


def test_program_solve():
    dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
        add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
    df = pd.DataFrame({
        'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
        'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
        'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
    })
    problem.read_data(df, covariates = ['X'])
    print(problem.solve())

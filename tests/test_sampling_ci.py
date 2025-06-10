from autobounds.autobounds.causalProblem import causalProblem, respect_to
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Q import Q
import pandas as pd
import numpy as np


dag = DAG("D -> Y")
df0 = pd.DataFrame({
    'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
    'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
})

df = pd.DataFrame({
    'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
    'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
    'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
})

def test_standard_solve():
    dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
    problem.read_data(df, covariates = ['X'])
#    print(problem.solve())
    print(problem.solve())


# def test_generate_samples():
#     dag = DAG("D -> Y")
#     problem = causalProblem(dag)
#     problem.read_data(df, covariates=['X'])
#     problem.generate_samples(n=10)
#     problem0 = causalProblem(dag)
#     problem0.read_data(df0)
#     problem0.generate_samples(n=10)



# def test_program_read_data_no_covariates():
#     problem = causalProblem(dag)
#     problem0 = causalProblem(dag)
#     with respect_to(problem0):
#         set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
#         read_data(df0)
#         print(solve())
#     with respect_to(problem):
#         set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
#         read_data(df0, inference = True)
#         generate_samples(n = 10)
#         calculate_ci()
#     print(problem.samples)
#     print(problem.lower_samples)


def test_program_read_data_covariates():
    dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    with respect_to(problem):
        set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
    problem.read_data(df0, inference = True)
#    print(problem.solve())
    print(problem.solve(ci = True, nsamples = 10))



# def test_program_calculate_ci():
#     dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
#     problem = causalProblem(dag)
#     with respect_to(problem):
#         set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
#         add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
#     df = pd.DataFrame({
#         'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
#         'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
#         'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
#     })
#     problem.read_data(df, covariates = ['X'])
#     np.random.seed(19103)
#     res0 = problem.calculate_ci(categorical = False, nsamples = 2)
#     res = ( np.quantile(res0[0], 0.025), np.quantile(res0[1], 0.975) )
#     assert res[0] > -0.55 and res[0] < -0.54
#     assert res[1] < 0.001


# def test_program_solve():
#     dag = DAG("D -> Y, U -> D, U -> Y", unob = "U")
#     problem = causalProblem(dag)
#     with respect_to(problem):
#         set_estimand(p('Y(D=1)=1') - p('Y(D=0)=1'))
#         add_assumption(p('Y(D=0)=0&Y(D=1)=1') - Q(0))
#     df = pd.DataFrame({
#         'X': [0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1],
#         'D': [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0],
#         'Y': [1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1]
#     })
#     problem.read_data(df, covariates = ['X'])
#     print(problem.solve())

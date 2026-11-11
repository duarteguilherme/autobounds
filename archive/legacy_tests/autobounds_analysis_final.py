import pandas as pd


from autobound.DAG import DAG
from autobound.causalProblem import causalProblem



def run_ate(problem):
    problem.set_ate('X', 'Y')
    program = problem.write_program()
    program.to_pip('/home/beta/test.pip')
#    return program.run_couenne()
    return program.run_pyomo('ipopt')

def run_late(problem):
    problem.set_estimand(
       problem.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') + 
       problem.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0', -1),
       div = problem.query('X(Z=1)=1&X(Z=0)=0'))
    program = problem.write_program()
    return program.run_couenne()



# DAG 2 -- no monotonicity / exclusion restriction
dag2_no_er = DAG()
dag2_no_er.from_structure("V -> Z, V -> X, Z -> X, Z -> W, Z -> Y, W -> Y, X -> Y, U -> X, U -> Y", unob = "U")
problem2_no_er = causalProblem(dag2_no_er)
#problem2_no_er.load_data('data/VZ.csv')
problem2_no_er.load_data('data/VZWXY.csv')


#run_ate(problem2_no_er)




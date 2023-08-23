# Script to calculate bounds for a case of measurement error

from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io


# Define DAG
dag = DAG()
dag.from_structure("X -> Y, Y -> S, U -> S, U -> Y", unob = "U")

# Define causal Problem
problem = causalProblem(dag)

# Add data, Set Estimand ATE, Add axioms of probability, and other constraints
problem.load_data('data/measurement_error.csv')
problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
problem.add_constraint(problem.query('S(Y=0)=1&S(Y=1)=0'))
problem.add_prob_constraints()

# Write program 
program = problem.write_program()
program.run_couenne(filename = 'results/measurement_error.csv')


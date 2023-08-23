from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io
import os


# Add causal model
dag = DAG()
dag.from_structure("X -> Y, Y -> R, X -> R")

# Start causal problem
problem = causalProblem(dag)

# Add two files of data
problem.load_data('data/manski1_1.csv', cond = ['R'])
problem.load_data('data/manski1_2.csv')
problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
problem.add_prob_constraints()

# Solve program
program = problem.write_program()


# Saving program
program.to_pip('results/nonresponse.pip')

#program.run_couenne(filename = 'results/nonresponse.csv')



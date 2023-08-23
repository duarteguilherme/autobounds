# Outcome Selection problem


from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io

# Loading DAG
dag = DAG()
dag.from_structure("Y -> S, X -> Y, U -> X, U -> Y", unob = "U")

# Loading causal Problem
problem = causalProblem(dag)

# Adding data, estimand (ATE), and axioms of probability
problem.load_data('data/selection_obsqty.csv')
problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
problem.add_prob_constraints()

# Generating optimization program
program = problem.write_program()
program.run_couenne(filename = 'results/selection.csv')



import pandas as pd
from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io
from copy import deepcopy
from autobound.Query import Query
#### Creating DAGs

# Mediation DAG
dag = DAG()
dag.from_structure("Z -> X, X -> Y, Z -> Y, U -> X, U -> Y", unob = "U")

problem = causalProblem(dag)
problem.load_data('data/iv.csv')
problem.add_prob_constraints()


ate_problem = deepcopy(problem)
cde_problem0 = deepcopy(problem)
cde_problem1 = deepcopy(problem)

ate_problem.set_ate('X','Y')
cde_problem0.set_estimand(
        cde_problem0.query('Y(X=1,Z=0)=1') - 
        cde_problem0.query('Y(X=0,Z=0)=1')
)
cde_problem1.set_estimand(
        cde_problem1.query('Y(X=1,Z=1)=1') - 
        cde_problem1.query('Y(X=0,Z=1)=1')
)


# Bounds for CDE, with Z forced to be 0
cde_problem0.write_program().run_couenne()
# Bounds for CDE, with Z forced to be 1
cde_problem1.write_program().run_couenne()



# Bounds for the NDE
nde_problem = deepcopy(problem)
# E(Y(Z=1, X(Z=0)) - Y(Z=0, X(Z=0))
nde_rh_query = nde_problem.query('Y(Z=0)=1')
# To write, lh_query,
# check values of
#     nde_problem.query('X(Z=0)=0') = X00 and X01
#     nde_problem.query('X(Z=0)=1') = X10 and X11
nde_part1 = nde_problem.query('Y(Z=1,X=1)=1')
nde_part2 = nde_problem.query('Y(Z=1,X=0)=1')
nde_lh_query1 = [ x for x in nde_part2 if 'X00' in x[1][0] or 'X01' in x[1][0] ]
nde_lh_query2 = [ x for x in nde_part1 if 'X10' in x[1][0] or 'X11' in x[1][0] ]
nde_query = Query(nde_lh_query1) + Query(nde_lh_query2) - nde_rh_query
nde_problem.set_estimand(nde_query)
nde_problem.write_program().run_couenne()

# Bounds for the NIE
nie_problem = deepcopy(problem)
# E(Y(Z=0, X(Z=1)) - Y(Z=0, X(Z=0))
nie_rh_query = nde_problem.query('Y(Z=0)=1')
# To write, lh_query,
# check values of
#     nie_problem.query('X(Z=1)=0') = X00 and X10
#     nie_problem.query('X(Z=1)=1') = X01 and X11
nie_part1 = nie_problem.query('Y(Z=0,X=1)=1')
nie_part2 = nie_problem.query('Y(Z=0,X=0)=1')
nie_lh_query1 = [ x for x in nie_part2 if 'X00' in x[1][0] or 'X10' in x[1][0] ]
nie_lh_query2 = [ x for x in nie_part1 if 'X01' in x[1][0] or 'X11' in x[1][0] ]
nie_query = Query(nie_lh_query1) + Query(nie_lh_query2) - nie_rh_query
nie_problem.set_estimand(nie_query)
nie_problem.write_program().run_couenne()





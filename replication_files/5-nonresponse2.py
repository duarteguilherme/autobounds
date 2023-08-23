from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io
import pandas as pd


# Wrangle data
dfx = pd.read_csv('data/manski2_obs_x_rx1.csv')
dfa = pd.read_csv('data/manski2_obs_rx_ry.csv')
dfa = dfa.groupby(['A']).sum().reset_index()
pa1 = dfa.loc[lambda a: a.A == 1]['prob'].iloc[0]
dfx['prob'] = dfx['prob'].div(pa1)
dfx = dfx[['X','prob']]

dfx.to_csv('data/manski2_only_x.csv', index = None)

# Prepare model
dag = DAG()
dag.from_structure("X -> Y, A -> B, Y -> B")

# Add problem
problem = causalProblem(dag)


# Load datafiles
problem.load_data('data/manski2_only_x.csv', optimize = True)
problem.load_data('data/manski2_obs_rx_ry.csv', optimize = True)
problem.load_data('data/manski2_obs_x_y_rx1_ry1.csv', optimize = True)
problem.load_data('data/manski2_obs_y_ry1.csv', optimize = True)

# Set estimand
problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
problem.add_prob_constraints()
program = problem.write_program()


# result = program.run_couenne(filename = 'results/nonresponse2.csv', theta = .02)

# Save program in pip file
# For this specific problem, we will be using SCIP
program.to_pip('results/nonresponse2.pip')

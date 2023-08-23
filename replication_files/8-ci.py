from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io
from copy import deepcopy
import pandas as pd
import numpy.random as rd

rd.seed(19015)


# First, we create functions to simulate data
# They will be simulated from the IV data using the weights with replacement

def simulate_data(n_sample):
    df = pd.read_csv('data/iv.csv')
    prob = list(df['prob'])
    df = df.drop('prob', axis = 1)
    df = (df
            .sample(n_sample, weights = prob, replace= True)
            .assign(prob = 1)
            .groupby(['Z','X','Y'])
            .sum().div(n_sample)
            .reset_index()
            )
    return df

def save_simul(n_sample):
    for i in range(1000):
        if i % 100 == 0:
            print(f'{i}', end = ',')
        simulate_data(n_sample).to_csv(f'results/cover_data/n{n_sample}/{i}.csv', index = None)
    print('\n')


# Saving simulations - n equal to 1000, 10000, 100000
save_simul(1000)
save_simul(10000)
save_simul(100000)



# Standard IV DAG
dag = DAG()
dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")


# Loading data, starting problem and solving for real bounds
import pandas as pd
pd.read_csv('data/iv.csv')
problem1 = causalProblem(dag)
problem1.load_data('data/iv.csv')
problem1.add_prob_constraints()
problem1.set_ate('X','Y')
real_value = problem1.write_program().run_pyomo('ipopt')
real_lb = real_value[0]
real_ub = real_value[1]
data_real = { 'real_ub': real_ub, 'real_lb': real_lb }



# Function to collect results, using ipopt as a optimization solver
def collect_result_ipopt(file, N):
    res_kl = run_kl(file, N, solver = 'ipopt')
    res_gaussian = run_gaussian(file, N, solver = 'ipopt')
    res_default = run_default(file, solver = 'ipopt')
    return { 
            'file': file,
            'ub_kl': res_kl[1],
            'lb_kl': res_kl[0],
            'ub_gauss': res_gaussian[1],
            'lb_gauss': res_gaussian[0],
            'ub_def': res_default[1],
            'lb_def': res_default[0]
            }

# Function to collect results using couenne
def collect_result(file):
    res_kl = run_kl(file)
    res_gaussian = run_gaussian(file)
    res_default = run_default(file)
    print(res_gaussian)
    return { 
            'file': file,
            'ub_kl': res_kl[1]['dual'],
            'lb_kl': res_kl[0]['dual'],
            'ub_gauss': res_gaussian[1]['dual'],
            'lb_gauss': res_gaussian[0]['dual']
            }


# Run bounds with KL-divergence confidence (described in the paper)
def run_kl(file, N, solver = 'couenne'):
    problem1 = causalProblem(dag)
    problem1.load_data_kl(file, N = N)
    problem1.add_prob_constraints()
    problem1.set_ate('X','Y')
    program1 = problem1.write_program()
    program1.run_pyomo('ipopt')
    if solver == 'couenne':
        return program1.run_couenne()
    else:
        return program1.run_pyomo('ipopt')



# Run point-estimated bounds 
def run_default(file, solver = 'ipopt', epsilon = 0.1):
    problem1 = causalProblem(dag)
    problem1.load_data(file)
    problem1.add_prob_constraints()
    problem1.set_ate('X','Y')
    program1 = problem1.write_program()
    if solver == 'couenne':
        return program1.run_couenne(epsilon = epsilon, verbose = False)
    else:
        return program1.run_pyomo('ipopt')

# Run Gaussian confidence bounds (described in the paper)
def run_gaussian(file, N, solver = 'ipopt', epsilon = 0.1):
    problem1 = causalProblem(dag)
    problem1.load_data_gaussian(file, N )
    problem1.add_prob_constraints()
    problem1.set_ate('X','Y')
    program1 = problem1.write_program()
    if solver == 'couenne':
        return program1.run_couenne(epsilon = epsilon, verbose = False)
    else:
        return program1.run_pyomo('ipopt')



# Running bounds for every simulated file
total_data = [ ]
for i in range(1000):
    print(i, flush = True)
    total_data.append({ **data_real, **(collect_result_ipopt(f'results/cover_data/n1000/{i}.csv', 1000))}) 
    total_data.append({ **data_real, **(collect_result_ipopt(f'results/cover_data/n10000/{i}.csv', 10000))}) 
    total_data.append({ **data_real, **(collect_result_ipopt(f'results/cover_data/n100000/{i}.csv', 100000))}) 


# Saving data
total_data_df = pd.DataFrame(total_data)
total_data_df.to_csv('results/ci_data.csv')



from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io
from copy import deepcopy
import timeit
import time 
import pandas as pd




# Standard IV DAG
dag = DAG()
dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")



comp_data = [ ]
# Problem 1 -- everything binary
#### Creating DAGs

datafile = io.StringIO('''X,Y,Z,prob
0,0,0,0.125
0,0,1,0.125
0,1,0,0.125
0,1,1,0.125
1,0,0,0.125
1,0,1,0.125
1,1,0,0.125
1,1,1,0.125''')

# Start
start_time1 = timeit.default_timer()

problem1 = causalProblem(dag)
problem1.add_prob_constraints()
problem1.load_data(datafile, optimize = False)
problem1.set_ate('X','Y')
program1 = problem1.write_program()
program1.run_pyomo('ipopt')
program1.run_couenne()

final_time1 = timeit.default_timer() - start_time1
comp_data.append({'problem': 'X binary, Y binary, Z binary', 'time': final_time1})
print(f'Problem 1 -- Time: {final_time1}')

# Problem 2 -- ternary instrument
#### Creating DAGs
prob = 1/12
datafile = io.StringIO(f'''X,Y,Z,prob
0,0,0,{prob}
0,0,1,{prob}
0,1,0,{prob}
0,1,1,{prob}
1,0,0,{prob}
1,0,1,{prob}
1,1,0,{prob}
1,1,1,{prob}
0,0,2,{prob}
0,1,2,{prob}
1,0,2,{prob}
1,1,2,{prob}''')

start_time2 = timeit.default_timer()
problem2 = causalProblem(dag, {'Z':3})
problem2.add_prob_constraints()
problem2.load_data(datafile, optimize = False)
problem2.set_ate('X','Y')
program2 = problem2.write_program()
program2.run_couenne()
final_time2 = timeit.default_timer() - start_time2
comp_data.append({'problem': 'X binary, Y binary, Z ternary', 'time': final_time2})
print(f'Problem 2 -- Time: {final_time2}')

# Problem 3 -- ternary treatment
#### Creating DAGs
prob = 1/12
datafile = io.StringIO(f'''Z,Y,X,prob
0,0,0,{prob}
0,0,1,{prob}
0,1,0,{prob}
0,1,1,{prob}
1,0,0,{prob}
1,0,1,{prob}
1,1,0,{prob}
1,1,1,{prob}
0,0,2,{prob}
0,1,2,{prob}
1,0,2,{prob}
1,1,2,{prob}''')


start_time3 = timeit.default_timer()
problem3 = causalProblem(dag, {'X':3})
problem3.add_prob_constraints()
problem3.load_data(datafile, optimize = False)
problem3.set_estimand(problem3.query('Y(X=1)=1') - problem3.query('Y(X=0)=1'))
program3 = problem3.write_program()
#program3.run_pyomo('ipopt')
# Check mistake with couenne
program3.run_couenne()
final_time3 = timeit.default_timer() - start_time3
comp_data.append({'problem': 'X ternary, Y binary, Z binary', 'time': final_time3})
print(f'Problem 3 -- Time: {final_time3}')



# Problem 4 -- ternary outcome
prob = 1/12
datafile = io.StringIO(f'''Z,X,Y,prob
0,0,0,{prob}
0,0,1,{prob}
0,1,0,{prob}
0,1,1,{prob}
1,0,0,{prob}
1,0,1,{prob}
1,1,0,{prob}
1,1,1,{prob}
0,0,2,{prob}
0,1,2,{prob}
1,0,2,{prob}
1,1,2,{prob}''')

start_time4 = timeit.default_timer()
problem4 = causalProblem(dag, {'Y':3})
problem4.add_prob_constraints()
problem4.load_data(datafile, optimize = False)
problem4.set_estimand(problem4.query('Y(X=1)=1') - problem4.query('Y(X=0)=1'))
program4 = problem4.write_program()
#program4.run_pyomo('ipopt')
program4.run_couenne()
final_time4 = timeit.default_timer() - start_time4
comp_data.append({'problem': 'X binary, Y ternary, Z binary', 'time': final_time4})
print(f'Problem 4 -- Time: {final_time4}')


# Problem 5 -- everything ternary
prob = 1/27
datafile = io.StringIO(f'''Z,X,Y,prob
0,0,0,{prob}
0,0,1,{prob}
0,0,2,{prob}
0,1,0,{prob}
0,1,1,{prob}
0,1,2,{prob}
0,2,0,{prob}
0,2,1,{prob}
0,2,2,{prob}
1,0,0,{prob}
1,0,1,{prob}
1,0,2,{prob}
1,1,0,{prob}
1,1,1,{prob}
1,1,2,{prob}
1,2,0,{prob}
1,2,1,{prob}
1,2,2,{prob}
2,0,0,{prob}
2,0,1,{prob}
2,0,2,{prob}
2,1,0,{prob}
2,1,1,{prob}
2,1,2,{prob}
2,2,0,{prob}
2,2,1,{prob}
2,2,2,{prob}''')

start_time5 = timeit.default_timer()
problem5 = causalProblem(dag, {'Z': 3, 'X': 3, 'Y':3})
problem5.add_prob_constraints()
problem5.load_data(datafile, optimize = False)
problem5.set_estimand(problem5.query('Y(X=1)=1') - problem5.query('Y(X=0)=1'))
program5 = problem5.write_program()
#program5.run_pyomo('ipopt')
program5.run_couenne()
final_time5 = timeit.default_timer() - start_time5
comp_data.append({'problem': 'X ternary, Y ternary, Z ternary', 'time': final_time5})
print(f'Problem 5 -- Time: {final_time5}')



comp_data2 = pd.DataFrame(comp_data)
comp_data2.to_csv('data/comp_data.csv', index = None)


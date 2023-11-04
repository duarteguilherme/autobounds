from autobounds.autobounds.causalProblem import causalProblem
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Program import change_constraint_parameter_value
from autobounds.autobounds.Query import Query
import io
import time
import pandas as pd


#from autobounds.causalProblem import causalProblem
#from autobounds.DAG import DAG
#from autobounds.Program import change_constraint_parameter_value
#from autobounds.Query import Query

def test_program_parse_whole_file():
    df = pd.DataFrame(
            {'Z': [0,0,0,0,1,1,1,1],
             'X': [0,0,1,1,0,0,1,1],
             'Y': [0,1,0,1,0,1,0,1],
          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
             })
    dag = DAG()
    dag.from_structure('Z -> X, X -> Y, U -> X, U -> Y', unob = 'U')
    pro = causalProblem(dag)
    pro.load_data(df)
    pro.add_prob_constraints()
    pro.set_ate('X','Y', cond = 'X(Z=1)=1&X(Z=0)=0')
    pro.add_constraint(pro.query('X(Z=1)=1&X(Z=0)=0') - Query(0.0001), '>=')
    program = pro.write_program()
    program.run_scip()
    program.res_scip
    program.track_result_scip()
    from autobounds.Program import get_final_bound_scip
    get_final_bound_scip('.lower.log')
    print(program.res_scip)


def test_optimizers():
    const = [['Y01'], ['-1', 'Y10'], ['X10', 'X11', 'Y11'], ['-1', 'objvar'], ['==']]
    print(change_constraint_parameter_value(const, 'Y10', 0.20))
    print(const)
    print(change_constraint_parameter_value(const, 'X11', 0.31))

def test_program_scip_infeasibility():
    dag = DAG()
    dag.from_structure("D -> Y")
    problem = causalProblem(dag)
    problem.set_estimand(problem.query('Y(D=1)=1') + problem.query('Y(D=0)=1', -1))
    problem.add_prob_constraints()
    problem.add_constraint(problem.query('Y(D=0)=0&Y(D=1)=1') - Query(2))
    z = problem.write_program()
    z.optimize_remove_numeric_lines()
    res = z.run_scip(maxtime = 5)
    assert res[0]['end'] == 0
    assert res[1]['end'] == 0

def test_program_scip_infeasibility2():
    df = pd.DataFrame(
            {'Z': [0,0,0,0,1,1,1,1],
             'X': [0,0,1,1,0,0,1,1],
             'Y': [0,1,0,1,0,1,0,1],
          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
             })
    dag = DAG()
    dag.from_structure('Z -> X, X -> Y, U -> X, U -> Y', unob = 'U')
    pro = causalProblem(dag)
    pro.load_data(df)
    pro.add_prob_constraints()
    pro.set_ate('X','Y', cond = 'X(Z=1)=1&X(Z=0)=0')
    pro.add_constraint(pro.query('X(Z=1)=1&X(Z=0)=0') - Query(0.0001), '>=')
    program = pro.write_program()
    program.run_scip()



def test_program_scip_time():
    dag = DAG()
    dag.from_structure("Z -> Y, X -> Y, U -> X, U -> Z", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_scip(maxtime = 5)
#
def test_program_scip():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,1.0
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_scip()
    assert res[0]['dual'] < 0.12
    assert res[1]['dual'] > 0.49
#    print(z.M_lower.getDualbound())
#    print(z.get_bounds_scip())
#    pot.writeProblem('/home/beta/pot.cip')
#

def test_program_pip_conf():
    dag = DAG()
    dag.from_structure("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''D,Y,prob
    0,0,0.25
    0,1,0.25
    1,0,0.25
    1,1,0.25
    ''')
    problem.set_ate('D','Y')
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_scip()


#
def test_program_parallel():
    dag = DAG()
    dag.from_structure("W -> X, W -> Y, W -> P, X -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,P,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_pyomo('ipopt', parallel = True, verbose = False)
    assert res[0] < -0.08
    assert res[0] < -0.08
    assert res[1] > -0.1
    assert res[1] > -0.1


def test_couenne_parse():
    dag = DAG()
    dag.from_structure("A -> Y, U -> A, U -> Y", unob = "U")
    problem = causalProblem(dag) 
    datafile = io.StringIO('''A,Y,prob
    0,0,0.13
    0,1,0.27
    1,0,0.2
    1,1,0.4''')
    problem.load_data(datafile)
    problem.set_ate('A','Y')
    program = problem.write_program()
    lower, upper, theta, epsilon = program.run_couenne()
    assert theta == 1
    assert epsilon == 0
    assert upper['primal'] < 0.54
    assert upper['primal'] > 0.52
    assert upper['dual'] < 0.54
    assert upper['dual'] > 0.52
    assert lower['primal'] < -0.46
    assert lower['primal'] > -0.48
    assert lower['dual'] < -0.46
    assert lower['dual'] > -0.48


def test_couenne_threshold():
    dag = DAG()
    dag.from_structure("A -> B, B -> Y, U -> A, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''A,B,Y,prob
    0,0,0,0.1
    0,0,1,0.1
    0,1,0,0.13
    0,1,1,0.1
    1,0,0,0.12
    1,0,1,0.15
    1,1,0,0.1
    1,1,1,0.2''')
    problem.load_data(datafile)
    problem.set_ate('A','Y')
    program = problem.write_program()
    result = program.run_couenne(theta = 0.4, epsilon = 1)



def test_numeric_lines():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile1 = io.StringIO('''X,A,prob
    0,1,0.230271252339654
    1,1,0.299527260157004''')
    datafile2 = io.StringIO('''Y,B,prob
    0,1,0.317930571936706
    1,1,0.176816814870372''')
    datafile3 = io.StringIO('''X,Y,A,B,prob
    0,0,1,1,0.087024102667622
    1,0,1,1,0.171021677876195
    0,1,1,1,0.0777025072996138
    1,1,1,1,0.0516439605484204''')
    datafile4 = io.StringIO('''A,B,prob
    0,0,0.362846349088114
    1,0,0.142406264104807
    0,1,0.107355138415228
    1,1,0.387392248391851''')
    dag = DAG()
    dag.from_structure("X -> Y, A -> B, Y -> B")
    problem = causalProblem(dag)
    problem.load_data(datafile1, optimize = True)
    problem.load_data(datafile2, optimize = True)
    problem.load_data(datafile3, optimize = True)
    problem.load_data(datafile4, optimize = True)
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.add_prob_constraints()
    program = problem.write_program()


def test_program_iv():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    b = z.run_pyomo(verbose=False)
    assert b[0] <= -0.48
    assert b[0] >= -0.52
    assert b[1] <= 0.52
    assert b[1] >= 0.48

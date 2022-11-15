from autobound.autobound.causalProblem import causalProblem
from autobound.autobound.DAG import DAG
from autobound.autobound.Program import change_constraint_parameter_value
import io



def test_optimizers():
    const = [['Y01'], ['-1', 'Y10'], ['X10', 'X11', 'Y11'], ['-1', 'objvar'], ['==']]
    print(change_constraint_parameter_value(const, 'Y10', 0.20))
    print(const)
    print(change_constraint_parameter_value(const, 'X11', 0.31))
    input("")

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
    for i in program.constraints:
        print(i)
        input("")
    print(program.run_pyomo('ipopt'))
    input("")



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

from autobound.autobound.DAG import DAG
from autobound.autobound.causalProblem import *
from autobound.autobound.Query import Query, clean_query
import io
from copy import deepcopy


def test_solve_kl():
    ns = 1000
    K = 8
    o = 0.125
    alpha = 0.05
    result = solve_kl_p(ns=ns, K = K, o = o, alpha = alpha)
    assert result[0] > 0.09
    assert result[1] < 0.17


def test_solve_gaussian():
    res = solve_gaussian(nr = 100, o = [0.25, 0.25, 0.25, 0.25] , alpha = 0.05)

def test_load_data_gaussian():
    datafile = io.StringIO('''X,Y,prob
    0,0,0.25
    0,1,0.25
    1,0,0.25
    1,1,0.25''')
    y = DAG()
    y.from_structure("U -> X, U -> Y", unob = 'U')
    x = causalProblem(y, {'X': 2})
    x.load_data_gaussian(datafile, N = 1000)
    x.add_prob_constraints()
    p00_problem, p01_problem, p10_problem, p11_problem = [deepcopy(x) for i in range(4) ]
    p00_problem.set_estimand(x.query('X=0&Y=0'))
#    p01_problem.set_estimand(x.query('X=0&Y=1'))
#    p10_problem.set_estimand(x.query('X=1&Y=0'))
#    p11_problem.set_estimand(x.query('X=1&Y=1'))
#    p00 = p00_problem.write_program().run_couenne()
#    p01 = p01_problem.write_program().run_couenne()
#    p10 = p10_problem.write_program().run_couenne()
#    p11 = p11_problem.write_program().run_couenne()
    p00_problem.write_program().to_pip('/home/beta/gauss.pip')


def test_load_data_kl():
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.125
    0,0,1,0.125
    0,1,0,0.125
    0,1,1,0.125
    1,0,0,0.125
    1,0,1,0.125
    1,1,0,0.125
    1,1,1,0.125''')
    datafile2 = deepcopy(datafile)
    y = DAG()
    y.from_structure("Z -> Y, X -> Y, U -> X, U -> Y", unob = 'U')
    x = causalProblem(y, {'X': 2})
    x.load_data_kl(datafile, N = 1000)
    assert -1 * x.constraints[-1][-2][0] == solve_kl_p(ns = 1000, K = 8, o = 0.125, alpha = 0.05)[1]



def test_add_constraints():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y, K -> X", unob = "U")
    x = causalProblem(y, {'X': 2})
    assert (1, 'Z0') in x.parameters
    x.set_p_to_zero(['Z0'])
    assert (0, 'Z0') in x.parameters
    x.add_constraint([(-0.15, ['1']), (-0.15, ['1']), (1, ['X1111']), (-1, ['X1111', 'Z1']), (2, ['X1111'])])
    assert [(-0.3, ['1']), (3, ['X1111']), (-1, ['X1111', 'Z1']), (1, ['=='])] in x.constraints
    x.add_constraint([(1, ['X1110']), (-1, ['X1110', 'Z1']), (-1, ['X1110'])])
    assert [(-1, ['X1110', 'Z1']), (1, ['=='])] in x.constraints

def test_check_constraints():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.load_data(datafile)
    x.add_prob_constraints()
    x.check_constraints()
    assert (0.5, ['X00.Y00', '1']) in x.constraints[0]

def test_set_ate():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    x.set_estimand(x.query('Y(X=1)=1&X=0') - x.query('Y(X=0)=1&X=0'), div = x.query('X=0'))
    x.set_ate('X','Y', cond = 'X=0')
    assert clean_query(x.constraints[-1]) == clean_query(x.constraints[-2])

def test_conditional_estimand():
    y = DAG()
    y.from_structure("X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    x.set_estimand(x.query('Y(X=1)=1') - x.query('Y(X=0)=1'), div = x.query('X=0'))
    assert Query(clean_query(x.constraints[-1])) ==  Query('X0.Y01') + Query('X1.Y01') - Query('X0.Y10') - Query('X1.Y10') - ( x.query('X=0') * Query('objvar') ) + Query('==')


def test_conditional_data():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.set_estimand(x.query('Y(X=1)=1') + x.query('Y(X=0)=1', -1))
    x.load_data(datafile, cond = ['X'])
    x.add_prob_constraints()
    z = x.write_program()
    assert 'objvar' in z.parameters
    assert 'X00.Y00' in z.parameters
    assert 'Z0' in z.parameters
    assert len(z.constraints) == 11
    assert z.constraints[1] ==  [['0.95', 'X00.Y00', 'Z0'], ['0.95', 'X00.Y01', 'Z0'], ['0.95', 'X01.Y00', 'Z0'], ['0.95', 'X01.Y01', 'Z0'], ['-0.05', 'X00.Y00', 'Z1'], ['-0.05', 'X00.Y01', 'Z1'], ['-0.05', 'X00.Y10', 'Z0'], ['-0.05', 'X00.Y10', 'Z1'], ['-0.05',   'X00.Y11', 'Z0'], ['-0.05', 'X00.Y11', 'Z1'], ['-0.05', 'X01.Y10', 'Z0'], ['-0.05', 'X01.Y11', 'Z0'], ['-0.05', 'X10.Y00', 'Z1'], ['-0.05', 'X10.Y01', 'Z1'], ['-0.05', 'X10.Y10', 'Z1'], ['-0.05', 'X10.Y11', 'Z1'], ['==']]


def test_causalproblem():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.set_estimand(x.query('Y(X=1)=1') + x.query('Y(X=0)=1', -1))
    x.load_data(datafile)
    x.add_prob_constraints()
    z = x.write_program()
    assert 'objvar' in z.parameters
    assert 'X00.Y00' in z.parameters
    assert len(z.constraints) == 10
#[['0.05'], ['0.5', 'X00.Y00'], ['0.5', 'X00.Y01'], ['0.5', 'X01.Y00'], ['0.5', 'X01.Y01'], ['==']]
    assert z.constraints[0] == [['X00.Y01'], ['X01.Y01'], ['X10.Y01'], ['X11.Y01'], ['-1', 'X00.Y10'], ['-1', 'X01.Y10'], ['-1', 'X10.Y10'], ['-1', 'X11.Y10'], ['-1', 'objvar'], ['==']]
    assert z.constraints[1] == [['0.5', 'X00.Y00'], ['0.5', 'X00.Y01'], ['0.5', 'X01.Y00'], ['0.5', 'X01.Y01'], ['-0.05'], ['==']]

def test_replace_first_nodes():
    assert replace_first_nodes([('Z0', 0.5), ('Z1', 0.5)], 
            (1, ['X00.Y10', 'Z0'])) == (0.5, ['X00.Y10', '1'])


def test_load_data():
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.125
    0,0,1,0.125
    0,1,0,0.125
    0,1,1,0.125
    1,0,0,0.125
    1,0,1,0.125
    1,1,0,0.125
    1,1,1,0.125''')
    datafile2 = deepcopy(datafile)
    y = DAG()
    y.from_structure("Z -> Y, X -> Y, U -> X, U -> Y", unob = 'U')
    x = causalProblem(y, {'X': 2})
    x.load_data(datafile)
    x.add_prob_constraints()
    x.constraints[3] == [(-1, ['0.25']), (1, ['X1.Y0001', 'Z1']), 
            (1, ['X1.Y0010', 'Z0']), (1, ['X1.Y0011', 'Z0']), 
            (1, ['X1.Y0011', 'Z1']), (1, ['X1.Y0101', 'Z1']), 
            (1, ['X1.Y0110', 'Z0']), (1, ['X1.Y0111', 'Z0']), 
            (1, ['X1.Y0111', 'Z1']), (1, ['X1.Y1001', 'Z1']), 
            (1, ['X1.Y1010', 'Z0']), (1, ['X1.Y1011', 'Z0']), 
            (1, ['X1.Y1011', 'Z1']), (1, ['X1.Y1101', 'Z1']), 
            (1, ['X1.Y1110', 'Z0']), (1, ['X1.Y1111', 'Z0']), (1, ['X1.Y1111', 'Z1'])] 
    y = DAG()
    y.from_structure("Z -> Y, U -> Z, X -> Y, U -> Y, U -> X", unob = "U")
    x = causalProblem(y, {'X': 2})
    x.load_data(datafile2)
    x.add_prob_constraints()
    x.constraints[1] == [(-1, ['0.125']), (1, ['X0.Y0000.Z1']), 
            (1, ['X0.Y0001.Z1']), (1, ['X0.Y0010.Z1']), (1, ['X0.Y0011.Z1']), 
            (1, ['X0.Y1000.Z1']), (1, ['X0.Y1001.Z1']), (1, ['X0.Y1010.Z1']), (1, ['X0.Y1011.Z1'])] 

def test_transform_constraint():
    assert transform_constraint([(1, ['X00.Y01']), (1, ['X01.Y01']), (1, ['X10.Y01']),
        (1, ['X11.Y01']), (-1, ['X00.Y10']), (-1, ['X01.Y10']), 
        (-1, ['X10.Y10']), (-1, ['X11.Y10']), (-1, ['1', 'objvar'])]) == [['X00.Y01'], 
                ['X01.Y01'], ['X10.Y01'], ['X11.Y01'], ['-1', 'X00.Y10'],
                ['-1', 'X01.Y10'], ['-1', 'X10.Y10'], ['-1', 'X11.Y10'], ['-1', 'objvar']]
    model = DAG()
    model.from_structure("D -> Y, D -> M, M -> Y, U -> Y, U -> M", unob = "U")
    problem = causalProblem(model)
    problem.set_p_to_zero([ x[1][0] for x in problem.query('M(D=0)=1&M(D=1)=0') ])
    problem.set_estimand(problem.query('Y(D=1)=1') + problem.query('Y(D=0)=1', -1),div = problem.query('M=1'))
    problem.constraints[-1]
    program = problem.write_program()
    assert program.constraints[-1] == [['M00.Y0010'], ['M00.Y0011'], ['M00.Y0110'], ['M00.Y0111'], ['M01.Y0001'], ['M01.Y0011'], ['M01.Y0101'], ['M01.Y0111'], ['M11.Y0001'], ['M11.Y0011'], ['M11.Y1001'], ['M11.Y1011'], ['-1', 'M00.Y1000'], ['-1', 'M00.Y1001'], ['-1', 'M00.Y1100'], ['-1', 'M00.Y1101'], ['-1', 'M01.Y1000'], ['-1', 'M01.Y1010'], ['-1', 'M01.Y1100'], ['-1', 'M01.Y1110'], ['-1', 'M11.Y0100'], ['-1', 'M11.Y0110'], ['-1', 'M11.Y1100'], ['-1', 'M11.Y1110'], ['-1', 'D0', 'M11.Y0000', 'objvar'], ['-1', 'D0', 'M11.Y0001', 'objvar'], ['-1', 'D0','M11.Y0010', 'objvar'], ['-1', 'D0', 'M11.Y0011', 'objvar'], ['-1', 'D0', 'M11.Y0100', 'objvar'], ['-1', 'D0', 'M11.Y0101', 'objvar'], ['-1', 'D0', 'M11.Y0110', 'objvar'], ['-1', 'D0', 'M11.Y0111', 'objvar'], ['-1', 'D0', 'M11.Y1000', 'objvar'], ['-1', 'D0', 'M11.Y1001', 'objvar'], ['-1', 'D0', 'M11.Y1010', 'objvar'], ['-1', 'D0', 'M11.Y1011', 'objvar'], ['-1', 'D0', 'M11.Y1100', 'objvar'], ['-1', 'D0', 'M11.Y1101', 'objvar'], ['-1', 'D0', 'M11.Y1110', 'objvar'], ['-1', 'D0', 'M11.Y1111', 'objvar'], ['-1', 'D1', 'M01.Y0000', 'objvar'], ['-1', 'D1', 'M01.Y0001', 'objvar'], ['-1', 'D1', 'M01.Y0010', 'objvar'], ['-1', 'D1', 'M01.Y0011', 'objvar'], ['-1', 'D1', 'M01.Y0100', 'objvar'], ['-1', 'D1', 'M01.Y0101', 'objvar'], ['-1', 'D1', 'M01.Y0110', 'objvar'], ['-1', 'D1', 'M01.Y0111', 'objvar'], ['-1', 'D1', 'M01.Y1000', 'objvar'], ['-1', 'D1', 'M01.Y1001', 'objvar'], ['-1', 'D1', 'M01.Y1010', 'objvar'], ['-1', 'D1', 'M01.Y1011', 'objvar'], ['-1', 'D1', 'M01.Y1100', 'objvar'], ['-1', 'D1', 'M01.Y1101', 'objvar'], ['-1', 'D1', 'M01.Y1110', 'objvar'], ['-1', 'D1', 'M01.Y1111', 'objvar'], ['-1', 'D1', 'M11.Y0000', 'objvar'], ['-1', 'D1', 'M11.Y0001', 'objvar'], ['-1', 'D1', 'M11.Y0010', 'objvar'], ['-1', 'D1', 'M11.Y0011', 'objvar'], ['-1', 'D1', 'M11.Y0100', 'objvar'], ['-1', 'D1', 'M11.Y0101', 'objvar'], ['-1', 'D1', 'M11.Y0110', 'objvar'], ['-1', 'D1', 'M11.Y0111', 'objvar'], ['-1', 'D1', 'M11.Y1000', 'objvar'], ['-1', 'D1', 'M11.Y1001','objvar'], ['-1', 'D1', 'M11.Y1010', 'objvar'], ['-1', 'D1', 'M11.Y1011', 'objvar'], ['-1', 'D1', 'M11.Y1100', 'objvar'], ['-1', 'D1', 'M11.Y1101', 'objvar'], ['-1', 'D1', 'M11.Y1110', 'objvar'], ['-1', 'D1', 'M11.Y1111', 'objvar'], ['==']]

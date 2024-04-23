from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.closedProblem import (
     closedProblem, get_cover_addition, is_inconclusive, get_cover_subtraction,
     Gset
)
from autobounds.autobounds.Parser import Parser
import pandas as pd
import io
from copy import deepcopy
from collections import Counter



dag = DAG()
dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
problem = closedProblem(dag)

def test_query():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    query_result = (Counter({'Z0': 1, 'Z1': 1}), Counter({'X00.Y01': 1, 'X00.Y11': 1, 'X01.Y01': 1, 'X01.Y11': 1, 'X10.Y01': 1, 'X10.Y11': 1, 'X11.Y01': 1, 'X11.Y11': 1})) 
    assert (problem.query('Y(X=1)=1') == query_result)


def test_solve_covering():
     problem.load_data('X,Y', 'Z=0')
     problem.load_data('X,Y', 'Z=1')
     problem.set_estimand(problem.query('Y(X=1)=1'))
     assert problem.solve_covering(0) is None # Test Step 1 -- index 0 must return 1
     problem.solve_covering(1)


def test_p_to_zero():
    problem.load_data('X,Y', 'Z=0')
    problem.load_data('X,Y', 'Z=1')


def test_comparisons_gset():
    x1 = Gset([0, 1, 1, 0]) # [1,2]
    x2 = Gset([0, 1, 0, 1]) # [1,3]
    x3 = Gset([0, 0, 1, 0]) # [2]
    x4 = Gset([0, 1, 1, 0]) # [1,2]
    x5 = Gset([0, 1, 1, 1]) # [1,2,3]
    assert (x1 >= x2) == False 
    assert (x1 >= x3) == True 
    assert (x1 >= x4) == True 
    assert (x1 >= x5) == False 
    assert (x1 > x2) == False 
    assert (x1 > x3) == True 
    assert (x1 > x4) == False
    assert (x1 > x5) == False 
    assert (x1 <= x2) == False 
    assert (x1 <= x3) == False 
    assert (x1 <= x4) == True 
    assert (x1 <= x5) == True
    assert (x1 < x2) == False 
    assert (x1 < x3) == False 
    assert (x1 < x4) == False 
    assert (x1 < x5) == True
    assert x1 / x2 == True 
    assert x1 / x3 == False 
    assert x1 / x4 == False
    assert x1 / x5 == False
    assert (x1 == x2) == False 
    assert (x1 == x3) == False 
    assert (x1 == x4) == True
    assert (x1 == x5) == False
    assert (x1 & x2) == True
    assert (x3 & x2) == False


def test_get_cover_addition():
    eset1 = Gset([0,1,0,0,1])
    eset2 = Gset([0,1,0,0,0])
    eset3 = Gset([0,1,1,1,0])
    eset4 = Gset([0,1,0,1,0])
    data1 = [ Gset([0,1,1,0,0]), Gset([0,0,1,1,0]), 
             Gset([0,0,0,1,1]), Gset([0,1,0,0,1]) ]
    assert get_cover_addition(eset1, data1) == [[3]]
    assert get_cover_addition(eset2, data1) == [[0], [3]] 
    assert get_cover_addition(eset3, data1) == [[0,1], [0,2]] 


def test_get_cover_subtraction():
    eset1 = Gset([0,0,1,0,0])    # 2
    data1 = [ Gset([0,1,1,1,0]), # 1,2,3
              Gset([1,0,1,1,0]), # 0,2,3
              Gset([0,1,0,0,0]), # 1
              Gset([0,1,1,1,1]) ] # 1,2,3,4
    # Notice that pieces 0 and 1 covers eset1. Piece 3 too, however, piece 0 is covered by piece 3
    # Now, Piece 0 minus Piece 3 is less than Piece 2 and still covers eset1
    atest1 = get_cover_subtraction(eset1, data1, [[0], [1]]) 
    assert atest1[0][0][0] == 0
    assert atest1[0][1][0] == 2
    data1.append(Gset([0,0,0,1,0])) # 3
    atest2 = get_cover_subtraction(eset1, data1, [[0], [1]]) 
    assert atest2[0][0][0] == 0
    assert atest2[0][1][0] == 2
    assert atest2[0][1][1] == 4
    # Now it has to return only [[[0]], [[2],[4]] ]
    # More tests required






def test_find_solutions():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Z')
    problem.set_estimand(problem.query('Y(X=1)=1'))
    problem.find_solutions()



def test_iv3():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag, {'Z': 3})
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Y,X', 'Z=2')
    problem.load_data('Z')
    problem.set_estimand(problem.query('Y(X=1)=1'))
    problem.find_solutions()





def test_find_solution_ate():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Z')
    problem.set_estimand(problem.query('Y(X=1)=1'))
    problem.find_bounds_subtraction(problem.query('Y(X=0)=1'))
    problem.read_solution()






# def test_iv4():
#     dag = DAG()
#     dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
#     problem = closedProblem(dag, {'Z': 4})
#     problem.load_data('Y,X', 'Z=0')
#     problem.load_data('Y,X', 'Z=1')
#     problem.load_data('Y,X', 'Z=2')
#     problem.load_data('Y,X', 'Z=3')
#     problem.load_data('Z')
#     problem.set_estimand(problem.query('Y(X=1)=1'))
#     problem.find_solutions()






# def test_iv5():
#     dag = DAG()
#     dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
#     problem = closedProblem(dag, {'Z': 5})
#     problem.load_data('Y,X', 'Z=0')
#     problem.load_data('Y,X', 'Z=1')
#     problem.load_data('Y,X', 'Z=2')
#     problem.load_data('Y,X', 'Z=3')
#     problem.load_data('Y,X', 'Z=4')
#     problem.load_data('Z')
#     problem.set_estimand(problem.query('Y(X=1)=1'))
#     problem.find_solutions()


 
def test_load_data():
    dag = DAG()
    dag.from_structure("Z -> X, X -> W, U -> X, U -> W", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('X,Y', 'Z=1')
    assert problem.data['X(Z=1)=0&Y(Z=1)=0'][0]['Z0'] == 1
    assert problem.data['X(Z=1)=0&Y(Z=1)=0'][1]['W01.X00'] == 1
    assert problem.data['X(Z=1)=1&Y(Z=1)=1'][1]['W11.X11'] == 1
    problem.load_data('X,Y', 'Z=0')
    assert (len(problem.data)) == 8
    # Test load_data iwth no do
    problem = closedProblem(dag)
    problem.load_data('Z,X,Y')
    assert problem.data['Z=0&X=0&Y=0'][0]['Z0'] == 1



def test_monotonicity():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Z')
    problem.set_estimand(problem.query('X(Z=0)=1&X(Z=1)=0'))
    problem.find_solutions()



def test_monotonicity_symmetric():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('Y,X', 'Z=0')
    problem.load_data('Y,X', 'Z=1')
    problem.load_data('Z')
    print(problem.query('X(Z=0)=1&X(Z=1)=0'))
    problem.set_estimand(problem.get_symmetric(problem.query('X(Z=0)=1&X(Z=1)=0')))
    problem.find_solutions()


def test_factorial():
    dag = DAG()
    dag.from_structure("A -> Y, B -> Y, U -> A, U -> B, U -> Y", unob = 'U')
    problem = closedProblem(dag)
    problem.load_data('Y', 'A=0,B=0')
    problem.load_data('Y', 'A=0,B=1')
    problem.load_data('Y', 'A=1,B=0')
    problem.load_data('Y', 'A=1,B=1')
    problem.load_data('A,B,Y')
    estimand = tuple([problem.query('Y(A=1)=1&B(A=1)=0')[0] + problem.query('Y(A=1)=1&B(A=1)=1')[0]])
    print(estimand)
    problem.set_estimand(estimand)
#    print(  problem.query('Y(A=1)=1&B(A=1)=1') + problem.query('Y(A=1)=1&B(A=1)=0'))

#    problem.set_estimand(problem.query('Y(A=1)=1&B(A=1)=1') + problem.query('Y(A=1)=1&B(A=1)=0'))
    problem.find_solutions()

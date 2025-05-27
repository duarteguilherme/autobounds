from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.causalProblem import causalProblem
from autobounds.autobounds.Parser import *
from autobounds.autobounds.Q import ( Query, clean_list, Q, 
                                     sub_list, add_list, 
                                     mul_list, compare_lists)
import pytest


model1 = DAG('D -> Y')
problem1 = causalProblem(model1)


def test_query_cond_zero():
    with pytest.raises(Exception, match="Condition cannot have probability zero"):
        q1 = Q(problem1.p('Y=1&D=1'), cond = problem1.p('Y=1') - problem1.p('Y=1'))
        q2 = Q(problem1.p('Y=1&D=1'), cond = Q(0))
    q3 = Q(problem1.p('Y=1&D=1'), cond = problem1.p('Y=1'))    
    print('\n')
    print(q3)

def test_add_sub_list():
    lst0 = [(2, ['X1.Y11'])]
    lst1 = [(1, ['X0.Y00']), (1, ['X1.Y11'])]
    lst2 = [(1, ['X0.Y00']), (-1, ['X1.Y11'])]
    lst3 = [(1, ['X0.Y00']), (-1, ['X0.Y00'])]
    lst4 = [(1, ['Z0', 'X0.Y00']), (1, ['Z0', 'X1.Y11'])]
    lst5 = [(1, ['Z0', 'X0.Y00']), (1, ['Z0', 'X1.Y11'])]
    assert clean_list(lst1) == sub_list(lst1, [])
    assert clean_list(lst0) == sub_list(lst1, lst2)
    assert clean_list(lst3) == sub_list(lst1, lst1)
    assert clean_list([]) == sub_list(lst4, lst5)
    # Test cases for add_query
    lst6 = [(1, ['Z0']), (1.5, ['X00'])]
    lst7 = [(1, ['Z0']), (1.5, ['X00'])]
    lst8 = [(2, ['Z0']), (3, ['X00'])]
    assert clean_list(lst8) == add_list(lst6, lst7)
    # Test combining terms with different scalars and multiplicative parts
    lst9 = [(1, ['Z0']), (1.5, ['X00'])]
    lst10 = [(0.5, ['Z0']), (2, ['X01'])]
    lst11 = [(1.5, ['Z0']), (1.5, ['X00']), (2, ['X01'])]
    assert clean_list(lst11) == add_list(lst9, lst10)
    # Test adding and subtracting combined
    lst12 = [(1, ['Z0']), (1.5, ['X00']), (2, ['X01'])]
    lst13 = [(1, ['Z0']), (1.5, ['X00'])]
    lst14 = [(2, ['X01'])]
    assert clean_list(lst14) == sub_list(lst12, lst13)

def test_mul_list():
    lst1 = [(2, ['X00', 'Z0']), (1, ['X01', 'Z1'])]
    lst2 = [(3, ['X00', 'Z0']), (1, ['X00', 'Z1'])]
    lst3 = [(6, ['X00', 'X00', 'Z0', 'Z0']), (2, ['X00', 'X00', 'Z0', 'Z1']), (3, ['X00', 'X01', 'Z0', 'Z1']), (1, ['X00', 'X01', 'Z1', 'Z1'])]
    assert mul_list(lst1, lst2) == lst3


def test_compare_identical_lists():
    lst1 = [(1.5, ['X00']), (2, ['X01']), (1, ['Z0'])]
    lst2 = [(1, ['Z0']), (1.5, ['X00']), (2, ['X01'])]
    assert compare_lists(lst1, lst2) is True

def test_compare_different_lists():
    lst1 = [(1.5, ['X00']), (2, ['X01']), (1, ['Z0'])]
    lst2 = [(1.5, ['X00']), (2, ['X01'])]
    assert compare_lists(lst1, lst2) is False

def test_compare_empty_lists():
    lst1 = []
    lst2 = []
    assert compare_lists(lst1, lst2) is True

def test_compare_one_empty_list():
    lst1 = [(1, ['Z0'])]
    lst2 = []
    assert compare_lists(lst1, lst2) is False

def test_compare_lists_with_redundant_terms():
    lst1 = [(1, ['Z0']), (1, ['Z0']), (2, ['X01'])]
    lst2 = [(2, ['X01']), (2, ['Z0'])]
    assert compare_lists(lst1, lst2) is True

def test_compare_lists_with_different_scalars():
    lst1 = [(1, ['Z0']), (2, ['X01'])]
    lst2 = [(1, ['Z0']), (3, ['X01'])]
    assert compare_lists(lst1, lst2) is False

def test_compare_lists_with_different_multiplicative_parts():
    lst1 = [(1, ['Z0']), (2, ['X01'])]
    lst2 = [(1, ['Z0']), (2, ['X02'])]
    assert compare_lists(lst1, lst2) is False

def test_query_more_letters():
    model1 = DAG()
    model1.from_structure('Dt -> Yt')
    problem1 = causalProblem(model1)
    problem1.p('Yt=1')


def test_query():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    # Test equality 
    assert x.p('Y=1') == x.p('Y=1')
    assert not (x.p('Y=1') == x.p('Y=0'))
    # Test subtraction and addition
    assert (x.p('Y=1') + x.p('Y=0', sign = -1)) == (x.p('Y=1') - x.p('Y=0'))


def test_types():
    assert Q('objvar') == Q([(1, ['objvar'])])
    assert Q(0.19) == Q([(0.19, ['1'])])
    assert Q(int('9')) == Q([(int(9), ['1'])])


def test_mul():
    assert (Q('X0.Y00') * Q('X1.Y11') * Q(0.29)) == Q([(0.29, ['X0.Y00', 'X1.Y11'])])

def test_clean_list():
    duplicated_query = clean_list([(1, ['X00.Y10', 'Z0']), (1, ['X00.Y10', 'Z0']) ] ) 
    unordered_query =  clean_list([(1, ['Z0','X00.Y11']), (1, ['X00.Y10', 'Z0']) ]  )
    zero_query = clean_list([(1, ['Z0','X00.Y11']), (-1, ['X00.Y11', 'Z0']) ]  )
    assert unordered_query[:] == [(1, ['X00.Y11', 'Z0']), (1, ['X00.Y10', 'Z0'])]
    assert duplicated_query[:] == [(2, ['X00.Y10', 'Z0'])]
    assert zero_query[:] == []

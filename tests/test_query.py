from autobound.autobound.DAG import DAG
from autobound.autobound.causalProblem import causalProblem
from autobound.autobound.Parser import *
from autobound.autobound.Query import Query, clean_query

def test_query():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    # Test equality 
    assert x.query('Y=1') == x.query('Y=1')
    assert not (x.query('Y=1') == x.query('Y=0'))
    # Test subtraction and addition
    assert (x.query('Y=1') + x.query('Y=0', -1)) == (x.query('Y=1') - x.query('Y=0'))


def test_types():
    assert Query('objvar') == Query([(1, ['objvar'])])
    assert Query(0.19) == Query([(0.19, ['1'])])
    assert Query(int('9')) == Query([(int(9), ['1'])])


def test_mul():
    assert (Query('X0.Y00') * Query('X1.Y11') * Query(0.29)) == Query([(0.29, ['X0.Y00', 'X1.Y11'])])

def test_clean_query():
    duplicated_query = Query(  [(1, ['X00.Y10', 'Z0']), (1, ['X00.Y10', 'Z0']) ] )
    unordered_query = Query(  [(1, ['Z0','X00.Y11']), (1, ['X00.Y10', 'Z0']) ] )
    zero_query = Query(  [(1, ['Z0','X00.Y11']), (-1, ['X00.Y11', 'Z0']) ] )
    duplicated_query.clean()
    unordered_query.clean()
    zero_query.clean()
    assert unordered_query[:] == [(1, ['X00.Y11', 'Z0']), (1, ['X00.Y10', 'Z0'])]
    assert duplicated_query[:] == [(2, ['X00.Y10', 'Z0'])]
    assert zero_query[:] == []

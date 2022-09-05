from autobound.autobound.DAG import DAG
from autobound.autobound.causalProblem import causalProblem
from autobound.autobound.Parser import *
from autobound.autobound.Query import Query, clean_query

def test_query():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    print(x.query('Y=1') + x.query('Y=0'))
    assert x.query('Y=1') + x.query('Y=0', -1) == x.query('Y=1') - x.query('Y=0')
#    x.set_estimand(x.query('Y(X=1)=1&X=0') + x.query('Y(X=0)=1&X=0', -1), div = x.query('X=0'))


def test_clean_query():
    duplicated_query = Query(  [(1, ['X00.Y10', 'Z0']), (1, ['X00.Y10', 'Z0']) ] )
    unordered_query = Query(  [(1, ['Z0','X00.Y11']), (1, ['X00.Y10', 'Z0']) ] )
    zero_query = Query(  [(1, ['Z0','X00.Y11']), (-1, ['X00.Y11', 'Z0']) ] )
    duplicated_query = clean_query(duplicated_query)
    unordered_query = clean_query(unordered_query)
    zero_query = clean_query(zero_query)
    assert unordered_query[:] == [(1, ['X00.Y11', 'Z0']), (1, ['X00.Y10', 'Z0'])]
    assert duplicated_query[:] == [(2, ['X00.Y10', 'Z0'])]
    assert zero_query[:] == []

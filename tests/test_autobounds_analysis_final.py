import pandas as pd


from autobound.autobound.DAG import DAG
from autobound.autobound.causalProblem import causalProblem
from autobound.autobound.Parser import Parser

#def test_parse_irre_expr():
#    dag = DAG()
#    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
#    test_p = Parser(dag)
#    print(test_p.parse_irreducible_expr('Y=0'))


def test_parse_expr():
    dag = DAG()
#    dag.from_structure("V -> Z, V -> X, Z -> X, Z -> W, Z -> Y, W -> Y, X -> Y, U -> X, U -> Y", unob = "U")
    dag.from_structure("X -> Y, U -> X, U -> Y", unob = "U")
    test_p = Parser(dag)
#    print(test_p.parse('P(Z=1)'))
#    print(test_p.parse('P(Y(X=1)=1)'))
    print(test_p.parse('P(X=1&Y=1)'))
#    print(test_p.parse('P(Y=1)'))

def test_collect_worlds():
    dag = DAG()
    dag.from_structure("V -> Z, V -> X, Z -> X, Z -> W, Z -> Y, W -> Y, X -> Y, U -> X, U -> Y", unob = "U")
    test_p = Parser(dag)
    test_p.collect_worlds("P(Y=0)") == {'': ['Y=0']}
    test_p.collect_worlds("P(Y(X=1)=0)") == {'X=1': ['Y=0']}
    test_p.collect_worlds("P(Y=0&X=0&X(Z=1)=0&V(Z=0,X=0)=0&W(Z=0,X=0)=1&Y(Z=0,X=0)=0)") == {'': ['Y=0', 'X=0'], 'Z=1': ['X=0'], 'Z=0,X=0': ['V=0', 'W=1', 'Y=0']}
   




#run_ate(problem2_no_er)




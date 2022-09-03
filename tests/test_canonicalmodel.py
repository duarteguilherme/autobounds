from autobound.autobound.DAG import DAG
from autobound.autobound.canonicalModel import canonicalModel
import numpy as np
from itertools import product
import math

def test_fromdag():
    x = canonicalModel()
    y = DAG()
    y.from_structure("Z -> Y, Z -> X, U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
    x.from_dag(y, {'X': 3})
    assert x.c_comp == set({frozenset({'Z'}), frozenset({'X', 'Y'})})
    assert x.number_parents == {'Z': 0, 'X': 1, 'Y': 2 }
    assert x.number_values == {'X': 3, 'Z': 2, 'Y': 2}
    assert x.number_canonical_variables == {'Z': 2, 'X': 9, 'Y': 64}
    assert x.parameters[0] == 'X00.Y000000'
    assert x.parameters[-1] == 'Z1'
    assert x.parameters[50] == 'X00.Y110010'



def test_get_functions():
    x = canonicalModel()
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U , Uy")
    x.from_dag(y, {'Y': 3})
    dag = DAG()
    dag.from_structure("V -> Z, V -> X, Z -> X, Z -> W, Z -> Y, W -> Y, X -> Y, U -> X, U -> Y", unob = "U")
    k = canonicalModel()
    k.from_dag(dag)
    assert x.get_functions(['X',0], [['Z',1]]) == ['X00','X10']
    assert k.get_functions(['V',0], [[]])  == ['V0']
    assert k.get_functions(['Z',0], [['V', 0]])  == ['Z00','Z01']
    


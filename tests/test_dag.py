from autobound.autobound.DAG import DAG
import numpy.random as rd
import pytest
import os

def test_dag_find_path():
    x = DAG()
    x.from_structure("B -> D,X -> Z, X -> Y, A -> X, A -> Z, Z -> Y, B -> A, B -> Y")
    paths = x.find_paths('B','Y')
    paths.sort()
    assert paths[0] == ['A', 'X', 'Y', True]
    assert x.find_inbetween('B','Y') == {'X', 'Z', 'A', 'Y', 'B'}

def test_dag_str():
    x = DAG()
    x.from_structure("U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
    assert x.V == set(('Y', 'X'))
    assert x.E == set((('Uy', 'Y'), ('X','Y'),('U','Y'),('U','X')))
    assert x.U == set(('Uy','U'))

def test_dag_find_algorithms():
    x = DAG()
    x.from_structure("U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
    y = DAG()
    y.from_structure("V -> Z, V -> X, Z -> X, W -> Y, Z -> W, X -> Y, U -> X, U -> Y", unob = "U")
    z = DAG()
    z.from_structure("A -> B, B -> C, C -> D, X -> Y, Y -> Z, Z -> D")
    assert x.find_parents('Y') == set(('Uy','X','U'))
    assert x.find_children('X') == set(('Y'))
    assert x.find_roots() == set(('Y'))
    assert x.find_first_nodes() == set(('X'))
    assert y.find_descendents(['Z']) == set(('X','W','Y'))
    assert z.find_ancestors(['D']) == set(('A','B','C', 'X', 'Y', 'Z'))

def test_dag_top_order():
    x = DAG()
    x.from_structure("""Z -> X, X -> Y, Z -> Y""")
    x.get_top_order()
    assert x.order[0] == 'Z'

def test_truncate():
    x = DAG()
    x.from_structure("""U -> X, X -> Y, U -> Y, Uy -> Y,
            X -> Z, Y -> Z, M -> Z, M -> A, Z -> A, Uma -> A,
            Uma -> M""", unob = "U , Uy, Uma")
    x.truncate('Z')
    assert 'Z' not in x.V
    assert ('M','Z') not in x.E

def test_c_comp():
    x = DAG()
    x.from_structure("""U -> X, X -> Y, U -> Y, Uy -> Y,
            X -> Z, Y -> Z, M -> Z, M -> A, Z -> A, Uma -> A,
            Uma -> M, U -> B, C -> D""", unob = "U , Uy, Uma")
    assert x.find_u_linked('X') == set({'X','Y', 'B'})
    assert frozenset({'X','B', 'Y'}) in x.find_c_components()

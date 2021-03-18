from causalid.causalid.DAG import DAG
import numpy.random as rd
import pytest
import os


def test_dag_str():
    x = DAG()
    x.from_structure("U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
    assert x.V == set(('Y', 'X'))
    assert x.E == set((('Uy', 'Y'), ('X','Y'),('U','Y'),('U','X')))
    assert x.U == set(('Uy','U'))

def test_dag_find_algorithms():
    x = DAG()
    x.from_structure("U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
    assert x.find_parents('Y') == set(('Uy','X','U'))
    assert x.find_children('X') == set(('Y'))
    assert x.find_roots() == set(('Y'))
    assert x.find_first_nodes() == set(('X'))

def test_dag_top_order():
    x = DAG()
    x.from_structure("""U -> X, X -> Y, U -> Y, Uy -> Y,
            X -> Z, Y -> Z, M -> Z, M -> A, Z -> A, Uma -> A,
            Uma -> M""", unob = "U , Uy, Uma")
    assert x.order[2] == set(('Z'))

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

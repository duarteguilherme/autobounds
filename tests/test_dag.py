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

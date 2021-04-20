from autobound.autobound.DAG import DAG
from autobound.autobound.SCM import SCM
import numpy as np
import pytest
import os



def test_scm_from_dag():
    dag = DAG()
    dag.from_structure("""U -> X, X -> Y, U -> Y, Uy -> Y,
            X -> Z, Y -> Z, M -> Z, M -> A, Z -> A, Uma -> A,
            Uma -> M""", unob = "U , Uy, Uma")
    scm = SCM()
    scm.from_dag(dag)
    assert 'U_X' in scm.U
    assert 'A' in scm.V
    assert len(scm.P) == 8
    assert 'U_M' in scm.F['M'][0]

def test_scm_sample():
    dag = DAG()
    dag.from_structure("""U -> X, X -> Y, U -> Y, Uy -> Y,
            X -> Z, Y -> Z, M -> Z, M -> A, Z -> A, Uma -> A,
            Uma -> M""", unob = "U , Uy, Uma")
    scm = SCM()
    scm.from_dag(dag)
    scm.sample_u(10)
    data = scm.draw_sample(intervention = {'X': 1})
    assert data['X'][0] == 1

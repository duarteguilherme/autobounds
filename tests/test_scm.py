from autobounds.DAG import DAG
from autobounds.SCM import SCM
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


def test_scm_preserves_multicharacter_variables_when_sampling():
    dag = DAG('Treatment -> Outcome')
    scm = SCM()
    scm.from_dag(dag)
    scm.F['Treatment'] = (('U_Treatment',), ())
    scm.F['Outcome'] = (('U_Outcome', 'Treatment'), (1,))
    scm.u_data = {'U_Treatment': np.array([0, 1, 0]),
                  'U_Outcome': np.array([0, 0, 1])}
    data = scm.draw_sample()
    assert data['Treatment'].tolist() == [0, 1, 0]
    assert data['Outcome'].tolist() == [0, 1, 1]

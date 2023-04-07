import numpy as np
from copy import deepcopy
import pandas as pd 

func_pool = [
        lambda a,b: a & b,
        lambda a,b: a | b,
        lambda a,b: a ^ b,
        lambda a,b:  ((a ^ b) - 1)*(-1)
        ]


def find_possible_functions(dag, v):
    """ For a specific variable, 
    return a possible f, based on the parents of v
    and functions from 0 to 3
    This functions is to be used i
    in the context of from_dag method in SCM data structure
    """
    pa = dag.find_parents(v)
    return ( 
            tuple([ 'U_' + v ] +
               [ 'U_' + x 
                   for x in pa if x in dag.U] +
               [  x 
                   for x in pa if x in dag.V]),
               tuple(x for x in np.random.randint(0,4,len(pa)))
               )

class SCM:
    """ 
    This class defines a structure for SCMs with four elements:
    self.u -> set size(Nu) -> contains unobserved variables
    self.v -> set size(Nv) -> contains observed variables
    self.f -> dict size(Nv) -> contains functions for each v variable
    self.p -> dict size(Nu) -> contains functions representing probability distributions over u
    
    Examples:
        scm = SCM()
        scm.set_u('U_X', lambda N: random.binomial(1, 0.75, N) )
        scm.set_u('U_XY', lambda N: random.binomial(1, 0.45, N) )
        scm.set_v('X', (lambda a: a, 'U_X')) 
        scm.set_v('Y', (lambda a,b: a, 'U_XY', 'X')) 
    
    """
    def __init__(self):
        self.U = set()
        self.V = set()
        self.F = {}
        self.P = {}
        
    def from_dag(self, dag):
        """ This function gets a DAG and returns 
        a SCM, with which that DAG is compatible. 
        The functions that control the behavior of each 
        V_i variable will be picked up from a pool of 
        'AND', 'OR', 'XOR' functions. 
    
        Examples: 
            scm = SCM()
             dag = DAG()
             dag.from_structure(('X -> Y, U_xy -> X, U_xy -> Y', unob = 'U_xy')
             scm.from_dag(dag)
        """
        self.dag = dag
        us = [ 'U_' + x for x in dag.U  ] + [ 'U_' + x for x in dag.V  ] 
        ps = np.random.uniform(0.2, 0.8, len(us))
        for i in range(len(ps)):
            self.set_u(us[i], ps[i])
        vs = dag.V
        for v in vs:
            self.set_v(v, find_possible_functions(dag, v))
    
    def set_u(self, u, p):
        self.U.add(u)
        self.P[u] = p
    
    def set_v(self, v, f):
        """ 
        f must be a tuple with the second element being a function numbers
        and the first one 
        the variables.
        So for example, (('U_v', 'U_m_v', 'V'), (0,2)) is a valid 
        function, to become a real function later
        """
        if len(f) < 2:
            raise Exception("Arg 'f' requires a tuple with at least two elements")
        if ( len(f[0]) != (len(f[1]) + 1) ):
            raise Exception("Number of functions must be equal to the number of variables minus 1")
        self.V.add(v)
        self.F[v] = f
    
    
    def sample_u(self, N):
        self.u_data = {}
        for u in self.U:
            self.u_data[u] = np.random.binomial(1, self.P[u], N)
    
    def draw_sample(self, intervention = {}):
        """ 
        Draw simulated sample according to the specified model.
        """
        dag = deepcopy(self.dag)
        data = pd.DataFrame(self.u_data)
        for key, values in intervention.items():
            dag.truncate(key)
            data[key] = values
        order = [ j for i in dag.get_top_order() for j in i]
        for v in order:
            data[v] = data[self.F[v][0][0]]
            for j in range(len(self.F[v][1])):
                data[v] = func_pool[self.F[v][1][j]](
                        data[v], data[self.F[v][0][j + 1]]
                        )
        return data





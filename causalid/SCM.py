from numpy import random


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
        self.u = set()
        self.v = set()
        self.f = {}
        self.p = {}
        
    def from_dag(self, dag = '', unob = ''):
        """ This function gets a DAG and returns 
        a SCM, with whom that DAG is compatible. 
        The functions that control the behavior of each 
        V_i variable will be picked up from a pool of 
        'AND', 'OR', 'XOR' functions. 
    
        Examples: 
             scm = SCM()
             scm.from_dag('X -> Y, U_xy -> X, U_xy -> Y', unob = 'U_xy')
        """
        pass
    
    def set_u(self, u, p):
        self.u.add(u)
        self.p[u] = p
    
    def set_v(self, v, f):
        """ 
        f must be a tuple with the first element being a function and the second one 
        a unobservable """
        if len(f) < 2:
            raise Exception("Arg 'f' requires a tuple with at least two elements")
        if not hasattr(f[0], "__call__"):
            raise Exception("Arg 'f' requires a first element of the tuple to be a function")
        self.v.add(v)
        self.f[v] = f
    
    
    def sample_v(self, N):
        self.m = np.array([ self.f_m(self.u_m[i], self.u_mb[i])
            for i in range(N) ]).reshape(-1)
        self.b = np.array([ self.f_b(self.u_b[i], self.u_mb[i])
            for i in range(N) ]).reshape(-1)
        self.y = np.array([ self.f_y(self.m[i], self.b[i],self.u_y[i])
            for i in range(N) ]).reshape(-1)
    
    def draw_sample(self, N):
        """ 
        Draw simulated sample according to the specified model.
        The size of the sample is specified through parameter N.
        The sample is a matrix (N, 3). The first column represents
        Y. The second and third ones represent M and B.
        """
        self.N = N
        self.sample_u(N)
        self.set_f()
        self.sample_v(N)
        




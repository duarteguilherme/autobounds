

class DAG:
    """ It defines a semi-Markovian DAG
    A semi-Markovian DAG is a structure with 
    three sets: V, representing observable variables,
    U, representing unobvservable variables,
    and E, representing edges from u \in U to 
    v \in V, or from V_i \in V to V_j \in V"""
    def __init__(self):
        self.V = set()
        self.U = set()
        self.E = set()
    
    def from_structure(self, decla):
        """
        get string and absorves the data 
        into the structure

        G <- DAG()
        G.from_structure('U -> X, X -> Y, U -> Y')
        """
        decla = decla.split(',')

    def add_v(self, v = ''):
        if v is not '' and v is not in self.V: 
            self.V.add(v)
    
    def set_u(self,u=''):
        if u is not '' and u is not in self.V: 
            self.V.remove(u)
            self.U.add(u)
    
    def add_e(self, a= '', b=''):
        if a is not '' and b is not '':
            if (a,b) is not in self.E:
                self.E.add((a,b))

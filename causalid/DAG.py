class DAG:
    """ It defines a semi-Markovian DAG
    A semi-Markovian DAG is a structure with 
    three sets: V, representing observable variables,
    U, representing unobvservable variables,
    and E, representing edges from u in U to 
    v in V, or from V_i in V to V_j in V
    """
    def __init__(self):
        self.reset_dag()
    
    def reset_dag(self):
        self.V = set()
        self.U = set()
        self.E = set()
    
    def from_structure(self, edges, unob = ''):
        """
        get string and absorves the data 
        into the structure
        G = DAG()
        G.from_structure('U -> X, X -> Y, U -> Y', unob = 'U')
        """
        edges = edges.split(',')
        for i in edges:
            edge = i.split('->')
            if len(edge) != 2:
                raise Exception("DAGs only accept edges with two variables. Verify input!")
            self.add_v(edge[0].strip())
            self.add_v(edge[1].strip())
            self.add_e(*edge)
        if unob != '':
            unob = unob.split(',')
            for i in unob:
                self.set_u(i.strip())
    
    def add_v(self, v = ''):
        if v == '' or ' ' in v:
            raise Exception("Method does not accept variable names with empty or space chars")
        if  v  not in self.V: 
            self.V.add(v)
    
    def set_u(self,u=''):
        if u == '' or ' ' in u:
            raise Exception("Method does not accept variable names with empty or space chars")
        if u in self.V: 
            self.V.remove(u)
            self.U.add(u)
    
    def add_e(self, a= '', b=''):
        a, b = a.strip(), b.strip()
        if a != '' and b != '' and a in self.V and b in self.V:
            if (a,b)  not in self.E:
                self.E.add((a,b))


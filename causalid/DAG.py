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
        edges = edges.replace('\r','').replace('\n','')
        unob = unob.replace('\r','').replace('\n','')
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
        self.get_topological_order()
    
    def find_parents(self, v):
        """ 
        Given a variable, find its parents
        """
        return set([ x[0] for x in self.E if x[1] == v.strip() ])
    
    def find_children(self, v):
        """ 
        Given a variable, find its children
        """
        return set([ x[1] for x in self.E if x[0] == v.strip() ])
    
    def find_roots(self):
        """ 
        Given a DAG, find all roots
        """
        v = self.V.copy()
        pa = set([ i[0] for i in self.E ])
        return v.difference(pa) # Roots cannot be parents
    
    def find_first_nodes(self):
        """ 
        Given a DAG, find all roots
        """
        v = self.V.copy()
        ch = set([ i[1] for i in self.E if i[0] not in self.U ])
        return v.difference(ch) # First nodes cannot be parents
    
    def get_topological_order(self):
        self.order = []
        first_nodes = self.find_first_nodes()
        v = self.V.copy()
        self.order.append(self.U) # Us are order 0
        if len(v) == 0:
            return None
        level = first_nodes.union(
                [ k for j in self.U for k in self.find_children(j) ])
        v = v.difference(level)
        # Set  level 1 (without U)
        self.order.append(level)
        while len(v) > 0:
            level = set([ k for j in level for k in self.find_children(j) 
                    if k in v])
            v = v.difference(level)
            self.order.append(level)
    
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


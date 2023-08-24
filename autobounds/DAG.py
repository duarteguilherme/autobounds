from functools import reduce
from math import pi
import networkx as nx
import matplotlib.pyplot as plt

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
      
    def plot(self):
        """
        Visualise a causal DAG using networkx
        """
        nodes = self.V.union(self.U)
        edges = self.E
        G = nx.DiGraph()    
        G.add_edges_from(edges)

        generations = [gen for gen in nx.topological_generations(G)] # topological ordering
        for layer, nodes in enumerate(generations): # create graph "layers" according to topological ordering
            for node in nodes:
                G.nodes[node]["layer"] = layer # add layers as node attributes

        pos = nx.multipartite_layout(G, subset_key="layer") # Compute the multipartite_layout using the "layer" node attribute
        
        # create curved edges to avoid parallel overlapping
        curved_edges = set()
        
        for edge in G.edges():
            source, target = edge
            for gen, nodes in enumerate(generations):
                if source in nodes:
                    ind1 = gen
                if target in nodes:
                    ind2 = gen
                    
            if source in self.U: # force disturbance links straight (for now)
                continue
            if ind2 - ind1 > 1: # curved edge if grandparents-grandchildren or greater
                curved_edges.add(edge)
                
        straight_edges = edges - curved_edges
        
        # label the nodes
        def label_format_latex(node_label):
            """ 
            Generate latex labels for the nodes
            """
            if len(node_label) == 1:
                return node_label
            else:
                big_letter = node_label[0].upper()
                subscript = node_label[1:].upper() # subscript after first letter
                return f"${big_letter}_{{{subscript}}}$"
        
        node_labels = {node:f"{label_format_latex(node)}" for node in G.nodes()}

        # draw the graph
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="white")
        nx.draw_networkx_labels(G, pos, ax=ax, labels = node_labels) 
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(straight_edges))
        arc_rad = pi / 9
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(curved_edges), connectionstyle=f'arc3, rad = {arc_rad}')
        fig.tight_layout()
        plt.show()
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
        self.get_top_order_with_u()
    
    def find_parents(self, v):
        """ 
        Given a variable, find its parents
        """
        return set([ x[0] for x in self.E if x[1] == v.strip() ])
    
    def find_parents_u(self, v):
        """ 
        Given a variable, find its parents -- only for V
        """
        return set([x for x in self.find_parents(v) if x in self.U ])
    
    def find_parents_no_u(self, v):
        """ 
        Given a variable, find its parents -- only for V
        """
        return set([x for x in self.find_parents(v) if x not in self.U ])
    
    def find_children(self, v):
        """ 
        Given a variable, find its children
        """
        return set([ x[1] for x in self.E if x[0] == v.strip() ])
    
    def find_descendents(self, v_set, no_v = True):
        """
        Given a list of variables, find all the descendents

        If no_v is set to True, method returns exclusive descendents, i.e., 
        all v' != v that are descendents of v 

        Not using recursion
        """
        desc = list(v_set)
        count = 0
        while count < len(desc):
            for j in self.find_children(desc[count]):
                if j not in desc:
                    desc.append(j)
            count += 1
        desc = set(desc)
        if no_v:
            for i in list(v_set):
                desc.remove(i)
        return desc
    
    def find_ancestors(self, v_set, no_v = True):
        """
        Given a list of variables, find all the ancestors

        If no_v is set to True, method returns exclusive ancestors, i.e., 
        all v' != v that are ancestors of v 

        Not using recursion
        """
        ancestors = list(v_set)
        count = 0
        while count < len(ancestors):
            for j in self.find_parents_no_u(ancestors[count]):
                if j not in ancestors:
                    ancestors.append(j)
            count += 1
        ancestors = set(ancestors)
        if no_v:
            for i in list(v_set):
                ancestors.remove(i)
        return ancestors
    
    def get_factors(self):
        """ If one has a regular DAG (not semi-Markovian),
        it returns the factorization
        """
        if len(self.U) > 0:
            raise Exception("This method only works for standard DAGs, not ADMGs.")
        factors = [ ]
        for v in self.get_top_order():
            factors.append((v,  self.find_parents(v)))
        return factors
    
    def find_paths(self, v1, v2):
        """ 
        Find all paths from v1 to v2
        Algorithm find_path: 
            Input: (v1, v2) 
            if v1 == v2:
                return v2
            Find all children ch of v1.
            If ch is empty:
                return False
            Else:
                return 
                [ ]
        """
        if v1 == v2:
            return [ True ]
        children = self.find_children(v1)
        if len(children) == 0:
            return [ False  ]
        else:
            return [ [ ch ] +  v  
                    if type(v) is list else [ ch ] + [ v ]   for ch in children 
                    for v in self.find_paths(ch,v2) if v != False ]
    
    def find_inbetween(self, v1, v2):
        """
            Find all variables in paths from v1 to v2
            Algorithm:
                self.find_paths from v1 to v2.
                get the union of all those variables and 
                remove v1 and v2
        """
        return set(filter(lambda x: x != True, 
            [v1] + reduce(lambda a, b: a + b, self.find_paths(v1,v2))))
        
    def visit(self, v):
        if v in self.order:
            return None
        if v in self.temp:
            raise Exception("Not a DAG!")
        self.temp.add(v)
        for j in self.find_children(v):
            self.visit(j)
        self.temp.remove(v)
        self.order.append(v)
        
    def find_u_linked(self,v):
        """ 
        find all the Vs linked to a specific v through 
        U variables
        """
        pa = self.find_parents(v)
        u_pa = pa.intersection(self.U)
        return set().union(*[self.find_children(i)  for i in u_pa ]) 
    
    def find_roots(self):
        """ 
        Given a DAG, find all roots
        For example, for a DAG X -> Y -> Z -> A, B -> Z -> A,
        A is a root. 
        The algoritm is: list all edges, and filter those that are not parents.
        """
        v = self.V.copy()
        pa = set([ i[0] for i in self.E ])
        return v.difference(pa) # Roots cannot be parents
    
    def truncate(self, T):
        T = set([ x.strip() for x in T.split(',') ])
        self.V = self.V.difference(T)
        e_to_remove = set([ x for x in self.E if x[1] in T ] )
        self.E = self.E.difference(e_to_remove)
    
    def find_c_components(self):
        c_comp = set()
        for v in self.V:
            cset = [ c for c in c_comp if v in c ]
            if len(cset) > 1:
                cset2 = [frozenset([ j for i in cset for j in i ]) ]
                for c in cset:
                    c_comp.discard(c)
                cset = cset2
            if len(cset) == 0:
                cset = [ frozenset([v]) ]
            c_comp.discard(cset[0])
            c_comp.add(cset[0].union(self.find_u_linked(v)))
        return c_comp
    
    def find_first_nodes(self):
        """ 
        Given a DAG, find all first nodes
        This is the opposite of find_roots...
        The algorithm is similar, but first nodes 
        cannot be children
        """
        v = self.V.copy()
        ch = set([ i[1] for i in self.E if i[0] not in self.U ])
        return v.difference(ch) # First nodes cannot be children
    
    def get_top_order_with_u(self):
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
    
    def get_top_order(self):
        """ DFS algorithm 
        Not elegant - must fix it later"""
        self.order = []
        self.temp = set() 
        V = self.V.copy()
        while len(V.difference(set(self.order))) > 0:
            self.visit(V.difference(set(self.order)).pop())
        self.order.reverse()
        return self.order
    
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


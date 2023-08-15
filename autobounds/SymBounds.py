from .canonicalModel import canonicalModel
from .Query import Query
from .Program import Program
from .DAG import DAG
from .Parser import Parser
from sympy import *
from itertools import product


list(product(['Z=0','Z=1'],['D=0','D=1'],['Y=0','Y=1']))


from autobounds.canonicalModel import canonicalModel
from autobounds.Query import Query
from autobounds.Program import Program
from autobounds.DAG import DAG
from autobounds.Parser import Parser



class symBounds:
    def __init__(self, dag, number_values = {}):
        self.canModel = canonicalModel()
        self.dag = dag
        self.canModel.from_dag(self.dag, number_values)
        self.Parser = Parser(dag, number_values)
        self.number_values = { }
        self.data = { }
        for i in dag.V:
            self.number_values[i] = number_values[i] if i in number_values.keys() else 2
        self.parameters = { x: Symbol(x, nonnegative = True) for x in self.canModel.parameters }
        self.first_nodes = self.dag.find_unconfounded_first_nodes() # First nodes are always intervened upon
        # First nodes are those unconfounded
        self.estimand = [ ]
        self.constraints = [ ]

    def set_first_node_intervention(self, restriction = [ ]):
        """
        This method defines all the interventions that must exist for all the data and estimands
        restriction parameter exist for cases of limited data 
        """
        self.first_node_inv = [ ','.join(i) for i in 
            list(
                product( *[ [ f'{j}={k}' 
                    for k in range(self.number_values[j])] 
                        for j in self.first_nodes])) ]
#        self.first_node_inv = [ f'({i})' for i in self.first_node_inv ]

    def query(self, expr, sign = 1):
        """ 
        Important function:
        This function is exactly like parse in Parser class.
        However, here it returns a constraint structure.
        So one can do causalProgram.query('Y(X=1)=1') in order 
        to get P(Y(X=1)=1) constraint.
        sign can be 1, if positive, or -1, if negative.
        """
        query1 = Query([ (sign, list(x)) for x in self.Parser.parse(expr) ])
        expr = expand('0')
        for i in query1:
            expr += expand(i[0]) * prod([self.parameters[j] for j in i[1]])
        return expr

    def solve(self):
        """ Solve the program finding max and min for the estimand
        """
        pass
    
    def find_max(self):
        """ Algorithm 
        The algorithm is based on the hypothesis that the maximum of a simple estimand
        can be expressed as a sum of strata. So one just have to see which 
        lists of principal strata contain the strata of the estimand.
        """
        pass

    def set_estimand(self,estimand):
        """
        Input: an expression similar to a constraint
        This algorithm there will 
        add estimand as a constraint with a new variable 
        objvar that will be added as a parameter.
        If the estimand is conditioned, then this condition 
        is multiplied by objvar, according to the algebraic formula.
        P(Y|X) = P(Y,X)/P(X) = objvar, then P(Y,X) - P(X) * objvar = 0
        """
        self.estimand = estimand
    
    def set_data(self, data = [ ], complete = True):
        if len(data) == 0:
            print('No data to be added!')
            return None
        if complete:
            for i in data:
                for j in i:
                    if j in self.first_nodes:
                        raise Exception(f'{j} is a first node. Data on unconfounded first nodes are not accepted. By default unconfounded first nodes are intervened upon.')
                    if j not in self.dag.V:
                        raise Exception(f'{j} does not exist in the graph')
                if len(self.first_node_inv) == 0:
                    for m in list(product( *[ [ f'{j}={k}'for k in range(self.number_values[j])] for j in i])):
                        self.data = {**self.data, 
                            **{'&'.join(m): self.query('&'.join(m)) } } 
                else:
                    for m in list(product( *[ [ f'{j}={k}'for k in range(self.number_values[j])] for j in i])):
                        for n in self.first_node_inv:
                            self.data = {**self.data, 
                            **{'&'.join(m) + '_' + n.replace(',',''): self.query('&'.join(m + f'({n})')) } }



dag = DAG()
dag.from_structure('Z -> D, D -> Y, U -> D, U -> Y', unob = 'U')
#dag.from_structure('D -> Y, U -> D, U -> Y', unob = 'U')
pro = symBounds(dag)


dag.find_first_nodes()

pro.set_first_node_intervention()
pro.first_node_inv

pro.
pro.first_node_inv

pro.set_data(data = ['DY'])

expand('D10_Y01 - D10_Y01')

pro.query('Y(D=1)=1')

pro.query('D(Z=0)=0&Y(Z=0)=0')

pro.data
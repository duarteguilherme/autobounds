from pyscipopt import Model
import numpy as np
from itertools import product
from causalid import SCM
from functools import reduce

##### IMPORTANT ***************************
# We have to create test to check if find_parents is always in order

func_r = lambda n: (np
        .array(list(product([0,1], repeat = 2**n)))
        .reshape(-1,*list(product([2], repeat = n))[0]))

def get_r_values(v, var, dag):
    pa = dag.find_parents_no_u(v)
    tot_pa = 2**len(pa)
    funcs = func_r(len(pa))
    return np.where(
            np.array([i[tuple([var[k] for k in pa])] 
                for i in funcs]) == var[v]
            )[0]

def expand_dict(dictio):
    return [ dict(zip(dictio.keys(), x)) 
            for x in product(*dictio.values())]

def mult(lista):
    return reduce(lambda a, b: a* b, lista)

def update_dict(a1, a2):
    for i in a2:
        if i in a1.keys():
            del a1[i]
    return dict(**a1,**a2)


class causalProgram(object):
    def __init__(self):
        self.parameters = dict()
        self.program = Model()
        self.obj_func = []
        self.constraints = []
    
    def from_dag(self, dag):
        self.dag = dag
        self.c_comp = self.dag.find_c_components()
        self.get_canonical_index(dag)
        for c in self.c_comp:
            for x in self.get_parameters(c):
                self.parameters[x] = self.program.addVar(x, vtype = "C") 
   
    def get_factorized_q(self, var):
        """ Receive values for variables 
        and return factorized version 
        of q_variables
        """
        factorization = []
        for c in self.c_comp:
            factorization.append('q-' + '.'.join(c) + '-' + 
                    '.'.join([ str(var[i]) for i in c])  )
        return factorization
    
    def get_expr(self, var = "", do = ""):
        """ Get an expression and 
        transform it to canonical form
        var identifies the variables 
        and do the interventions
        for example: P(Y=1|do(X=1)) can 
        become 
        program.define_expression(var = "Y=1", do = "X=1")
        The algorithm works by getting a list 
        of parameters and removing the unnecessary ones
        """
        if ( var == "" ):
            raise Exception("Error: specify v")
        var = dict([ (v[0].strip(), int(v[1])) 
            for v in [ v.split('=') for v in var.split(',') ] ])
        if ( do != ""):
            do = dict([ (v[0].strip(), int(v[1])) 
                for v in [ v.split('=') for v in do.split(',') ] ])
        rest_var = self.dag.V.difference(set(var.keys()))
        var_list = [ dict(**var, **dict(zip(rest_var, k)))
                for k in product([0,1], repeat = len(rest_var)) ]  
        expanded_var = expand_dict(self.get_q_index(var_list[0]))
        factorized = [ self.get_factorized_q(update_dict(j,do)) for i in var_list 
                for j in expand_dict(self.get_q_index(i)) ]
        factorized = [ mult([  self.parameters[j] for j in x ]) for x in factorized ]
        return factorized
     
    def get_q_index(self,var):
        """ Var here is a dictionary 
        For the model, all the variables must 
        be represented.
        For example, program.get({'X': 1, 'Y': 0})
        """
        q_index = {}
        self.var = var
        if  set(var.keys()) != self.dag.V:
            raise Exception("Error: provide values for all the variables")
        for v in var.keys():
            q_index[v] = list(get_q_values(v, var, dag))
        return q_index
    
    def get_parameters(self, c):
        q_values = product(*[ [ str(x) for x in range(2**(1+self.cn_index[v]))] for v in c ])
        params = tuple(['q-' + '.'.join(c) + '-' + '.'.join(q) for q in q_values])
        return params
    
    def get_canonical_index(self, dag):
        self.cn_index = {}
        for v in self.dag.V:
            self.cn_index[v] = len(
                    [i for i in self.dag.find_parents(v) 
            if i not in self.dag.U ])

        

#from pyscipopt import Model,quicksum
import numpy as np
from itertools import product
from functools import reduce


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
    def __init__(self, sense = "minimize"):
        self.parameters = dict()
#        self.program = Model()
        self.obj_func = []
        self.constraints = []
        self.sense = sense
    
    def from_dag(self, dag):
        self.dag = dag
        self.c_comp = self.dag.find_c_components()
        self.get_canonical_index()
        for c in self.c_comp:
            for x in self.get_parameters(c):
                self.parameters[x] = self.program.addVar(x, vtype = "C",
                        lb = 0, ub = 1) 
   
    def set_obj(self, expre):
        """
        Input: an expression written in pyscipopt format
        No output. Add objective into self.program
        """
        self.objvar = self.program.addVar(name = "objvar",
                vtype = "C", lb = None, ub = None)
        self.program.setObjective(self.objvar, self.sense)
        self.program.addCons(self.objvar == expre)
    
    def add_prob_constraints(self):
        """ 
        No input
        Add all axioms of probability constraints
        """
        c_components = [ '_'.join(x.split('_')[0:-1]) for x in self.parameters.keys() ]
        c_components = set(c_components)
#        for c in c_components:
#            self.program.addCons(
#                    quicksum([ self.parameters[p]
#                        for p in self.parameters if p.startswith(c) ]) == 1)
    
    def get_factorized_q(self, var):
        """ Receive values for variables 
        and return factorized version 
        of q_variables
        """
        factorization = []
        for c in self.c_comp:
            factorization.append('q_' + '.'.join(c) + '_' + 
                    '.'.join([ str(var[i]) for i in c])  )
        return factorization
    
    def get_response_expr(self, var={}):
        """ Differently from get_expr,
        this gets probabilities over response variables,
        and return equivalent expression.
        For example, P(Ry=1) self.get_response_expr({"Y":3}), will 
        return a sum of q_xy, for a c-component {X,Y}
        (This expression accepts only one c-component)
        """
        relevant_c = [ c for c in self.c_comp 
                if any([ i in c for i in var.keys()] )  ]
        if len(relevant_c) > 1:
            raise Expcetion("More than one c-component assigned.")
        relevant_c = relevant_c[0]
        parameters = []
        for v in relevant_c:
            if v in var.keys():
                parameters.append([var[v]])
            else:
                parameters.append(list(range(2**(1+self.cn_index[v]))))
        parameters = [ [ str(j) for j in i] for i in product(*parameters) ]
        parameters = [ 'q_' + '.'.join(relevant_c) + '_' + '.'.join(i) 
                for i in  parameters] 
        return parameters
    
    def get_expr(self, var = "", do = ""):
        """ Input: a probabilistic expression over V expression
        Output: canonical form expression
        for example: P(Y=1|do(X=1)) can 
        become 
        program.define_expression(var = "Y=1", do = "X=1")
        """
        if ( var == "" ):
            raise Exception("Error: specify v")
        var = dict([ (v[0].strip(), int(v[1])) 
            for v in [ v.split('=') for v in var.split(',') ] ])
        if ( do != ""):
            do = dict([ (v[0].strip(), int(v[1])) 
                for v in [ v.split('=') for v in do.split(',') ] ])
        else:
            do = {}
        rest_var = self.dag.V.difference(set(var.keys()))
        var_list = [ dict(**var, **dict(zip(rest_var, k)))
                for k in product([0,1], repeat = len(rest_var)) ]  
        expanded_var = expand_dict(self.get_q_index(var_list[0]))
        factorized = [ self.get_factorized_q(j) for i in var_list 
                for j in expand_dict(self.get_q_index(i, do)) ]
        factorized = [ mult([  self.parameters[j] for j in x ]) for x in factorized ]
        return factorized
     
    def get_q_index(self,var, do = {}):
        """ 
       Input: ocurrence of u. program.get_q_index({'Z':1, 'X':0, 'Y':1})
       Output: index. {'Z':[2,3], 'X':[0,1], 'Y': [2,3]}
        All the variables must 
        be represented.
        For example, program.get({'X': 1, 'Y': 0})
        """
        q_index = {}
        if  set(var.keys()) != self.dag.V:
            raise Exception("Error: provide values for all the variables")
        for v in var.keys():
            if v not in do.keys():
                q_index[v] = list(get_r_values(v, update_dict(var.copy(),do), self.dag))
            else:
                q_index[v] = list(get_r_values(v, var, self.dag))
        return q_index
    
    def get_parameters(self, c):
        """ 
        Input: c-component (frozenset)
        Output: tuple of all parameters of this c-component
        """
        q_values = product(*[ [ str(x) for x in range(2**(2**self.cn_index[v]))] for v in c ])
        # Only for binaries -> 2**(2**n)
        params = tuple(['q_' + '.'.join(c) + '_' + '.'.join(q) for q in q_values])
        return params
    
    def get_canonical_index(self):
        """
        Create self.cn_index 
        with canonical index for each variable in the DAG
        """
        self.cn_index = {}
        for v in self.dag.V:
            self.cn_index[v] = len(
                    [i for i in self.dag.find_parents_no_u(v) ])
    
    def check_indep(self, c):
        """
        In a certain c-component 'c',
        check for possible independencies among 
        response variables
        ------
        Input: c-component
        Output: independent response variable tuples
        """
        c = list(c)
        if len(c) < 3:
            return []
        res = []
        for i in range(len(c)-1):
            for j in range(i+1, len(c)):
                if len(
                        self.dag.find_parents_u(c[i]).intersection(
                        self.dag.find_parents_u(c[j])
                        )) == 0:
                    res.append({c[i],c[j]})
        return res
    
    def add_indep(self, var):
        """ 
        Input: Var
        This method will be called by add_indeps in order 
        to simplify code. 
        Independences for particular values will be added as constraints
        """
        keys = list(var.keys())
        cons1 = []
#        for i in [0,1]:
#            cons1.append(quicksum([ self.parameters[k]
#                    for k in 
#                    self.get_response_expr({keys[i]: var[keys[i]]}) ] ))
#        cons2 = quicksum([ self.parameters[k]
#                for k in 
#                self.get_response_expr(var) ])
        self.program.addCons(cons1[0]*cons1[1] - cons2 == 0)
    
    def add_rest_indep(self, indep):
        indep = list(indep)
        elem_1 = 2**(1+self.cn_index[indep[0]])
        elem_2 = 2**(1+self.cn_index[indep[1]])
        for i in range(elem_1):
            for j in range(elem_2):
                self.add_indep({indep[0]: i, indep[1]: j})
    
    def add_indep_constraints(self):
        """ For each components, check independencies 
        among variables and add them as constraints
        to the model. 
        """
        indeps = []
        for c in self.c_comp:
            indeps = indeps + self.check_indep(c)
        for i in indeps:
            print(i)
            self.add_rest_indep(i)
    






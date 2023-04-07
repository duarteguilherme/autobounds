#from autobound.DAG import DAG
import numpy as np
from itertools import product
import math



numbers = '0123456789abcdefghijklmnopqrstuvxwyz'

def convert_to_system(number, system):
    """ 
    Converts from decimal system to 
    any system one prefers
    """
    result = ''
    while number > 0:
        remainder = number % system
        number = int(number / system)
        result = numbers[remainder] + result
    return result

def get_range_number_system(end, system):
    """ 
    Generates range for one particular number system 
    -- end is the final value of a range, 
    for instance, range(10),
    -- system  refers to the number system, 2 (binary), 3 (ternary), etc
    """
    digit_range = math.ceil(math.log(end, system))
    return [ str(convert_to_system(i, system)).zfill(digit_range) for i in range(end) ]


def get_index(vec, number_values, total):
    if len(vec) == 0:
        return None
    index = 0
    r = total
    for i, k in enumerate(vec):
        r = r // number_values[i] 
        index += k * r
    return index


def filter_params(ind, params, v):
    return [ m for m in params if m[ind  + 1] == str(v[1]) ] 


class canonicalModel():
    """
    Class that can be related to a DAG, with 
    all the stratification of this DAG.
    Independent classes can be constructed, however.

    self.parameters will indicate all the parameters separated by c components.
    self.iso_params will indicate all the parameters separated by variables
    """
    def __init__(self):
        # Parameters will be modelled as lists. 
        # One must be careful to avoid repetition of elements and 
        # lack of order
        # The best way of solving this problem would be 
        # to use ordered sets rather than lists
        self.parameters = list()
        self.iso_params = {}
    
    def find_index(self, orde, value, ):
        pass

    def get_functions(self, v, data):
        """
        The purpose of this method is very simple.
        Suppose you want to know all the functions causing V = 1,
        and you have some data about some of X's parents, 
        for instance, Y=1, X=0. This method will return all the functions 
        allowing this transformation.

        INPUT: 1) v (target variable): [ 'V', 1] 
        2) data: [['Y',1], ['X',0]]
        OUTPUT: list of parameters satisfying those conditions

        STEP 1: get all parameters from iso_params for letter v
        STEP 2: for elements in data that are parents, get index of their presence in v params
        STEP 3: filter params that satisfy v[1] in those index.
        Given parents in format
        [ [A, 1], [B,2] ]
        return all values of v that satisfy those rules
        """
        params = self.iso_params[v[0]] # STEP 1
        total = len(params[0]) - 1
        parents = list(self.dag.find_parents_no_u(v[0]))
        if len(parents) == 0: # If there are no parents, so v returns v
            v[1] = str(v[1])
            return [''.join(v)]
        parents.sort()
        parents_data = { k[0]: k[1] for k in data }
        parents_data_keys = parents_data.keys()
        index_list = [ ]
        for i in parents:
            if i in parents_data_keys:
                index_list.append([i, [ parents_data[i] ]])
            else:
                index_list.append([i, list(range(self.number_values[i])) ])
#        index_list_pa = [ k[0] for k in index_list ] 
        number_values = [ self.number_values[k[0]] for k in index_list ] 
        index_list = list(product(*[ k[1] for k in index_list]))
        for k in index_list:
            params = filter_params(get_index(k,number_values,total), params, v)
        return params

    def from_dag(self, dag, number_values = {}):
        """
        This method acoplates a DAG to a causalProblem.
        By doing this, it already defines all the parameters of this problem,
        by converting the DAG to a Canonical Model. 
        The algorithm is:
            for c in c-components:
                for k in c.variables:
                    push all values of k
                list all different parameters for c
        -- var_card indicates the number of values for each variable. 
        If DAG has three variables, X, Y, Z, such that X and Y
        have 3 different values and Z has four values, then we would have to 
        add the argument number_values = {'X':3, 'Y':3, 'Z': 4}. If no information 
        about a variable is provided, then it is considered binary.
        """
        self.dag = dag
        self.c_comp = self.dag.find_c_components() 
        self.set_number_parents()
        self.number_values = number_values # number_values refers to 
        # the number of values of the variable in the original model,
        # not the number of canonical values, which depends on the 
        # number of parents
        for v in self.dag.V:
            if v not in number_values.keys():
                self.number_values[v] = 2
        self.set_parameters()
    
    def set_number_parents(self):
        """
        Generate self.number_parents 
        a dictionary with number of parents of each variable in V.
        Unobservable are not counted.
        """
        self.number_parents = {}
        for v in self.dag.V:
            self.number_parents[v] = len(
                    [i for i in self.dag.find_parents_no_u(v) ])
    
    def set_parameters(self):
        self.number_canonical_variables = {}
        # Needs order for c_comp --- alphanumeric
        c_comp = list(self.c_comp)
        c_comp = [ list(c) for c in c_comp ]
        list(map(lambda a: a.sort(), c_comp))
        c_comp.sort()
        self.iso_params = { }
        for v in self.dag.V:
            # To find the number of possible functions
            # Find first the number of ordered pairs for the independent variables,
            # |A|, which is the product of the values of each variable
            # Secondly, find te number of values for the dependent, contained 
            # in self.number_values, |B|
            # Finally, the number of canonical variables is 
            # |B|^|A|
            self.number_canonical_variables[v] = int( 
                    pow(
                        # B
                        self.number_values[v],
                        # A
                        np.prod(
                            tuple(( self.number_values[i] 
                            for i in self.dag.find_parents_no_u(v) ))
                        )
                    )
                )
            self.iso_params[v] = [ v + i for i in 
                    get_range_number_system(self.number_canonical_variables[v], 
                    self.number_values[v]) ]
        for c in c_comp:
            self.parameters += list(
                    product(*[ [ x + a for a in get_range_number_system(
                            self.number_canonical_variables[x],
                            self.number_values[x]) ]  for x in c ]
                        )
                    )
        self.parameters = [ '.'.join(x) for x in self.parameters ]
        self.parameters = list(set(self.parameters)) # Removing duplicated els
        self.parameters.sort() # Sorting parameters


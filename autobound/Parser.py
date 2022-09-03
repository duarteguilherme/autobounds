from .canonicalModel import canonicalModel
import numpy as np
from itertools import product
from functools import reduce
from copy import deepcopy




def find_vs(v,dag):
    ch = dag.find_children(v)
    if len(ch) == 0:
        return v
    else: 
        return [ find_vs(k, dag) for k in ch ]

def intersect_tuple_parameters(par1, par2):
    """
    Get two parameters, for instance,
    ('X0.Y0100', '') and ('X0.Y0100', 'Z1')
    and returns if they interesect.
    Empty strings are assumed to interesect with 
    everything.
    """
    if len(par1) != len(par2):
        raise Exception('Parameters have no same size')
    for i, el in enumerate(par1):
        if par1[i] != par2[i] and par1[i] != '' and par2[i] != '':
            return False
    # Next loop mixes both par1, par2, if they intersect
    par = [ ]
    for i in range(len(par1)):
        if par1[i] == '':
            if par2[i] == '':
                par.append('')
            else:
                par.append(par2[i])
        else:
            par.append(par1[i])
    return tuple(par)



def add2dict(dict2):
    def func_dict(dict1):
        res = {a: b for a,b in dict1.items() }
        for c,d in dict2.items():
            res[c] = d
        return res
    return func_dict

def intersect_expr(expr1, expr2, c_parameters):
    """
    For each element of each expression, 
    they have to be compared according to the c_components they are
    Example: [('Z1', 'X00.Y0100'), ('Z1', 'X00.Y1100')] and 
    [('W01.K1000', 'Z1'), ('W01.K1001', 'Z0').
    Output must be 
    """
    c_expr1 = [ [ list(set(c).intersection(set(k))) for c in c_parameters ] for k in expr1 ]
    c_expr2 = [ [ list(set(c).intersection(set(k))) for c in c_parameters ] for k in expr2 ]
    c_expr1 = [ tuple([ x[0] if len(x) != 0 else '' for x in c ])  for c in c_expr1 ] 
    c_expr2 = [ tuple([ x[0] if len(x) != 0 else '' for x in c ])  for c in c_expr2 ] 
    #res = list(set(c_expr1).intersection(set(c_expr2)))
    res = [ intersect_tuple_parameters(i,j) for i in c_expr1 for j in c_expr2 ]
    res = [ x for x in list(set(res)) if x ] 
    #res = [ tuple([ x for x in c if x != '' ]) for c in res ]
    return res 



def get_c_component(func, c_parameters):
    # Input: func is a list of found parameters. For example, [Z0, Y1000]
    # Input: c_parameters a list of list of all parameters of all c-components
    # Output: transformed func in terms of c_components
    c_flag = [ [ p for p in cp ] for cp in c_parameters 
        if any([x in p for x in func for p in cp]) ] 
    func_flag = []
    for c in c_flag:
        res = c.copy()
        for k in func:
            if not any([ k in x for x in c]):
                continue
            res = [ x for x in res if k in x ]
        func_flag.append(res)
    func_flag = list(product(*func_flag))
    return func_flag



def search_value(can_var, query, info):
    """
    Input:
        a) can_var: a particular canonical variable
        For instance, '110001'
        b) query: for a particular variable, determines 
        which value is being looked for. For instance,
        query = '1'
        c) info --- info is a list of parent values.
        For instance, if there are two parents X and Z 
        in alphanumeric order, such that, X has 3 values
        and Z, 2, then info = (3,2).
    Output: it returns a list with all the possibilities 
    the order of parents are alphanumeric
    [ (0,0), (0,1), ... ]
    ----------
    Algorithm: transforms values to a matrix, reshape according 
    to info, and then, with argwhere, returns all the important indexes.
    """
    array = np.array(list(can_var)).reshape(info)
    return np.argwhere(array == query)


def clean_irreducible_expr(expr):
    """
    This function will work as preprocess step 
    in Parser.parse_irreducible_expr.
    It gets an irreducible expression such as Y(x=1,Z=0)=0,
    and transforms it into a tuple with vars and values,
    for instance, 
    ( ['Y', 0], [ ['X', 1], ['Z', 0]])
    """
    expr = expr.strip()
    if ( '(' in expr and ')' not in expr ) or ( ')' in expr and '(' not in expr ):
        raise NameError('Statement contains error. Verify brackets!')
    if '(' in expr:
        do_expr = expr.split('(')[1].split(')')[0]
        do_expr = [ x.strip().split('=') for x in do_expr.split(',') ]
        do_expr = [ [ x[0].strip(), int(x[1].strip()) ] for x in do_expr ]
        main_expr = [ expr.split('(')[0].strip(),
                int(expr.split(')')[1].split('=')[1].strip()) ]
    else:
        main_expr = [ x.strip() for x in expr.split('=') ]
        main_expr = [ main_expr[0], int(main_expr[1]) ]
        do_expr = [ ]
    return (main_expr, do_expr)



 
class Parser():
    """ Parser 
    will include a DAG and a canonicalModel
    It will translate expressions written for DAGs 
    in terms of canonicalModels and vice-versa
    """
    def __init__(self, dag, number_values = {}):
        self.dag = dag
        self.canModel = canonicalModel()
        self.canModel.from_dag(self.dag, number_values = number_values)
        self.c_parameters = deepcopy([ [ k 
            for k in self.canModel.parameters if list(c)[0] in k ] 
            for c in self.canModel.c_comp ] )
        
    def parse_expr(self, world, expr):
        """
        It gets a whole expression in terms of a world and returns 
        the equivalence in terms of a canonical model

        Step 1) If there is intervention, original model must be truncated.
        For instance, a graph with Z -> X -> Y, with do(X = 1), 
        we must have a DAG with X forced to be 1. 
        
        Step 2) From the expression, select all relevant variables. 
        Model will have to include all those variables, as well as 
        their ancestors. 
        For instance, for a graphg Z -> X -> Y, if we query P(X = 1),
        we will have to select X and Z. Y can be discarded.
        They have to be put in topological order.
        
        Step 3) 
        """
        dag = deepcopy(self.dag)
        # STEP 1 -- truncate and remove do vars from main
        do_expr = [ i.split('=') for i in world.split(',') ]
        if do_expr != [['']]: # Clean do_expr 
            do_expr = [ [i[0], int(i[1]) ]for i in do_expr ]
        else:
            do_expr = list()
        do_var = [ i[0]  for i in do_expr ] 
        main_expr = [ i.split('=') for i in expr if i[0] not in do_var ]
        main_var = [ i[0] for i in main_expr ] 
        if  len(main_var) > len(set(main_var)): # Check if one is querying intersection such as Y = 1 and Y = 0
                return [] 
        dag.truncate(','.join([ x[0] for x in do_expr ]))
        # STEP 2 --- Get variable and its ancestors in topological order
        ancestors = list(dag.find_ancestors(main_var, no_v = False))
        all_var = [ i for i in dag.get_top_order() if i in ancestors ]
        
        # STEP 3 -- Translate from prob to can_prob
        # can_prob will be list of tuples. The first element of the tuple 
        # is a list with data including all the values of variables before.
        # For instance, for Z -> X -> Y,
        # if we are getting the functions from X to Y, we need to remember to include 
        # data for each piece of Z (1 or 0, if binary).
        # The second element is the list of parameters until that moment.
        #
        # An important detail of this step is that data for get_functions must be forced 
        # to include variables and values of do.
        can_prob = [ ([], [] ) ]
        for i in all_var:
            can_prob_next = [ ]
            if i in main_var:
                var_entry = [ b for b in main_expr if b[0] == i ][0]
                var_entry[1] = int(var_entry[1])
                for j in can_prob:
                    can_prob_next.append(
                            (
                            j[0] + [ var_entry ],
                            j[1] + [ self.canModel.get_functions(var_entry.copy(), j[0] + do_expr) ]
                            ))
                can_prob = can_prob_next
            else:
                for m in range(self.canModel.number_values[i]):
                    for j in can_prob:
                        var_entry = [i, m ]
                        can_prob_next.append(
                                (
                                j[0] + [ var_entry],
                                j[1] + [ self.canModel.get_functions(var_entry.copy(), j[0] + do_expr) ]
                                ))
                can_prob = can_prob_next
        
        # STEP 3.5 -- Get complete list of parameters
        can_param = [ list(product(*x[1])) for x in can_prob ] 
        can_param = [ j for i in can_param for j in i ]  
        
        # STEP 4 --- From parameters get c-components
        funcs = [ a for k in can_param for a in get_c_component(list(k), self.c_parameters) ]
        return funcs
    
    def collect_worlds(self, expr):
        """ 
        Gets an expr of variables and divide them according to different worlds.

        For instance, X=1,Y=1,X(Z=1)=1...
        X=1,Y=1 belong to the same worlds, but X(Z=1)=1 is a different world
        """
        exprs = expr.split('&')
        dict_expr = {}
        for i in exprs:
            j = i.split(')')
            if len(j) == 1:
                try:
                    dict_expr[''].append(i)
                except:
                    dict_expr[''] = [ i ]
                continue
            k = j[0].split('(')
            try:
                dict_expr[k[1]].append(k[0] + j[1])
            except:
                dict_expr[k[1]] = [ k[0] + j[1] ]
        return dict_expr


    def parse(self, expr):
        """
        Input: complete expression, for example P(Y(x=1, W=0)=1&X(Z = 1)=0)
        Output: a list of canonical expressions, representing this expr 
        -----------------------------------------------------
        Algorithm:
            STEP 1) Separate expr into exprs, according to different worlds.
            STEP 2) Run self.parse_expr on each of those exprs.
            STEP 3) Collect the interesection of those expressions
        """
        expr = expr.strip() 
        expr = expr.replace('P(', '', 1)[:-1] if expr.startswith('P(') else expr
        expr = expr.replace('P (', '', 1)[:-1] if expr.startswith('P (') else expr
        expr = expr.replace(' ','')
        exprs = self.collect_worlds(expr)
        exprs = [ self.parse_expr(i,j) for i,j in exprs.items() ]
        exprs = reduce(lambda a,b: intersect_expr(a,b, self.c_parameters), exprs)
        exprs = [ tuple(sorted([i for i in x if i != '' ]))  for x in exprs ] # Remove empty ''
        return sorted(exprs)
    

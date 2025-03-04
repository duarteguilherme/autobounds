from .canonicalModel import canonicalModel
from .Query import Query
import numpy as np
from itertools import product
from functools import reduce
from copy import deepcopy



def lsconcat(a,b):  # A concatenation function to replace empty lists
    if not (len(a) > 0 and len(a[0]) > 0):
        return b
    elif not (len(b) > 0 and len(b[0]) > 0):
        return a
    else:
        return a + b

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


def factor_by_c_component(path, c_comp_base, c_parameters):
    """
    The input will be a path in terms of simple strata,
    but we need the same path in terms of multiplication of c-component strata.

    For instance, 
    if the input is [[Z0, Z1], [X00, X01], [Y00, Y01]]
    We need the out put [[Z0,Z1], [X00.Y00, ...]
    """
    c_copy = c_parameters.copy()
    for p in path:
        for index, base in enumerate(c_comp_base):
            if p[0] in base:
                c_copy[index] = [ k 
                        for k in c_copy[index] 
                        for j in p if j in k ]
    return tuple(c_copy)



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
        # c-parameters must be in topological order 

    def translate(self, main_expr, do_expr, ancestors):
        """ Output (all_paths): a list of tuples, where each tuple represents a path that satisfies the quantity in main_var.
         
         In particular, each tuple (a path) will have two entries: the first one is a list with the path for 
         the original model, and the second one the path for the canonical model. In other words, the original 
         expression and the translated version.
        
         In the first list, the parameters are represented also as lists with two entries: first is the name of the variable and second is the value
         In the second list, you have the list of principal strata/response variables.
          
         The algorithm for translation is a BFS one.
         It starts with all the root ancestors ( layer 1 ).
         
         The number of canonical parameters of the root ancestor
         is identical to the number of values their variable can assume 
         in the original model
         
         For every descendent variable (layers - 2, 3, ...), the parameters 
          will depend to the value of their parents in the original model
        
         Consider this example: Let the model Z -> X, Z -> Y, X -> Y,
         This model has only root Z, with two values: Z0 and Z1
        
         Now consider X, if X = 0, you have to consider two cases:
         a) If Z = 1, then X will include the canonical parameters X00 (X[Z=0]=0,X[Z=1]=0) and X10 (X[Z=0]=1,X[Z=1]=0),
         b) If Z = 0, then X will include the canonical parameters X00 (X[Z=0]=0,X[Z=1]=0) and X01 (X[Z=0]=0,X[Z=1]=1),
         
         For considering cases of Y, then the same is done, but now using values of Z and X in the original model
        
         An important detail of this step is that data for get_functions must be forced 
         to include variables and values of do.
        
         the tuples in can_prob will include also the 
         As I noticed, maybe it's better to store every search
         """
        main_var = list(main_expr.keys())
        do_var = [ i[0] for i in do_expr ]
        all_paths = [ ([[]], [[]] ) ] # It starts with an empty path
        for var in ancestors: # In other words, each ancestor has to be considered in topological order
            if var in main_var: # If var was referred in the main_expr, so one does not need to consider all possibilities (all_poss)
                all_poss = [ int(main_expr[var]) ]
            else: # If is not referred, then it needs to consider all possitibilities
                all_poss = range(self.canModel.number_values[var])
            all_paths_new = [] # Before second loop
            # Each instance of poss will become a path in all_paths,
            # so at the end use all_paths = all_paths_new
            for path in all_paths:
                for poss in all_poss:
                    all_paths_new.append( 
                        (
                        lsconcat(path[0], [[var, poss ]]),
                        lsconcat(path[1],  [ self.canModel.get_functions( [var, poss ], 
                                lsconcat(path[0], do_expr) ) ] ) 
                        )
                    )
            all_paths = all_paths_new # After the second loop
        return all_paths
    
    def p(self, expr, sign = 1):
        """ 
        Important function:
        This function is exactly like parse in Parser class.
        However, here it returns a constraint structure.
        So one can do causalProgram.p('Y(X=1)=1') in order 
        to get P(Y(X=1)=1) constraint.
        sign can be 1, if positive, or -1, if negative.
        """
        return Query([ (sign, list(x)) for x in self.parse(expr) ])
    
    def is_active(self, expr = '', ind = '', dep = ''):
        """ This method collects all the strata with respect to a direct edge from ind to dep
        
        Notice that one can introduce X -> Y as expr rather than separately ind = X and dep = Y

        This method returns a List of strata in string. 
        This is different from a Query with the same strata.
        """
        if expr != '':
            ind, dep = [i.strip() for i in expr.split('->')]
        parents = self.dag.find_parents_no_u(dep)
        if ind not in parents:
            return []
        rest = [ i for i in parents if i != ind] 
        edge_values = list( 
            product(*( 
                [ range(0,self.canModel.number_values[dep]) ]*self.canModel.number_values[ind]   
                ) ) )
        edge_values = [ i for i in edge_values if not all([j == i[0] for j in i])]
        query = Query([])
        if len(rest) == 0: # If ind is the only parent, then you 
            for i in edge_values:
                query += self.p('&'.join([ f'{dep}({ind}={index})={j}'
                                 for index, j in enumerate(i)
                                 ]) ) 
            return [ i[1][-1] for i in query ]
        rest_entries = list(product(*[ range(self.canModel.number_values[i]) for i in rest ]))
        rest_entries = [ ','.join([  f'{j}={i[index]}' 
                                   for index, j in enumerate(rest)]) 
                        for i in rest_entries ]
        for i in edge_values:
            for k in rest_entries:
                query += self.p('&'.join([ f'{dep}({ind}={index},{k})={j}'
                                 for index, j in enumerate(i)
                                 ]) )
        return [ i[1][-1] for i in query ]
    
    def parse_expr(self, world, expr, complete = False):
        """
        Input: a probability expression for only one world  
        Output: the equivalent term in parameters of the canonical model

        world argument indicates possible interventions
        expr indicates the expression to be evaluated
        complete indicates if one wants to cover only parameters that ancestors to the relevant variables, 
            or every parameter that is not a descendent of the relevant variables. 
            The first case is the default (complete = False)

        Step 1) If there is intervention, original model must be truncated.
        For instance, a graph with Z -> X -> Y, with do(X = 1), 
        we must have a DAG with X forced to be 1, i.e. Z   X1 -> Y
        
        Step 2) From the expression, select all relevant variables. 
        Model will have to include all those variables, as well as 
        their ancestors. 
        For instance, for a graph Z -> X -> Y, if we query P(X = 1),
        we will have to select X and Z. Y can be discarded.
        They have to be put in topological order.
        
        Step 3) Use the translation algorithm defined above in method .translate

        Step 4) Return the parameters in terms of c-components
        For instance, if X <-> Y, then X0 would not be a parameter, but X0Y00
        """
        dag = deepcopy(self.dag)
        # STEP 1 -- truncate and remove do vars from main
        do_expr = [ i.split('=') for i in world.split(',') ]
        if do_expr != [['']]: # Clean do_expr 
            do_expr = [ [ i[0], int(i[1])]   for i in do_expr ]
        else:
            do_expr = [ ]
        do_var = [ i[0] for i in do_expr ]
        main_expr = [ i.split('=') 
            for i in expr if i[0] not in do_var ]
        main_var = [ i[0] for i in main_expr ]
        if  len(main_var) > len(set(main_var)): # Check if one is querying intersection such as Y = 1 and Y = 0
            return [] 
        main_expr = dict(main_expr)
        main_expr = { k: int(val) for k, val in main_expr.items() }
        dag.truncate(','.join([ x[0] for x in do_expr ]))
        # main_var indicates the variables we want to identify
        
        # STEP 2 --- Get variable and its ancestors in topological order
        # Ancestors include the variables
        # To find complete paths -- rather than find_ancestors, one has to use -- NOT SURE if valid
        ancestors = list(dag.find_ancestors(main_var, no_v = False))
        ancestors = [ i for i in dag.get_top_order() if i in ancestors ] # Put them in order

        # STEP 3 -- Translate from prob to can_prob
        all_paths = self.translate(main_expr, do_expr, ancestors)     
        if complete:
            # If complete is True, it returns factors in terms of paths
            # Eventually this will become default
            c_comp_base = [  set([j for i in c for j in i.split('.') ]) 
                for c in self.c_parameters]
            all_paths = [ factor_by_c_component(path[1], c_comp_base, self.c_parameters)
                    for path in all_paths ] 
            return all_paths

        # Change the structure of all_paths
        all_paths = [ list(product(*x[1])) for x in all_paths ] 
        all_paths = [ j for i in all_paths for j in i ]  

        # STEP 4 --- Get parameters in terms of  c-components parameters
        funcs = [ a for k in all_paths for a in get_c_component(list(k), self.c_parameters) ]
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


    def parse(self, expr, complete = False):
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
        exprs = [ self.parse_expr(i,j, complete) for i,j in exprs.items() ]
        if complete:
            # Now it suffices to do the intersection of each expr by factor
            # Check if every path has the same path
            if len(set([ len(i) for k in exprs for i in k])) != 1:
                raise Exception("Paths with different lengths!")
            exprs = reduce(lambda path1, path2: [ tuple( list(set(p1[i]) & set(p2[i]) ) 
                                                for i in range(len(p1)))
                for p1 in path1 for p2 in path2 ],
                   exprs) 
            exprs = [ i for i in exprs if not any([ len(k) == 0 for k in i ]) ]
            return exprs
        exprs = reduce(lambda a,b: intersect_expr(a,b, self.c_parameters), exprs)
        exprs = [ tuple(sorted([i for i in x if i != '' ]))  for x in exprs ] # Remove empty ''
        return sorted(exprs)
    

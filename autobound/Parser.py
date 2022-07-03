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


def get_funcs(parser, funcs, v, do_to_dict):
    funcs_output = [ ]
    print(f"Size of funcs is: {len(funcs)}")
    for f in funcs:
        for r in parser.filter_functions(v, do_to_dict(f[1])):
#            print(r)
            funcs_output.append( (r[0] + ',' + f[0], r[1] ) )
    return funcs_output

 
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
        """
        dag = deepcopy(self.dag)
        do_expr = [ i.split('=') for i in world.split(',') ]
        do_var = [ i[0]  for i in do_expr ] 
        main_expr = [ i.split('=') for i in expr if i[0] not in do_var ]
        main_var = [ i[0] for i in main_expr ] 
        print(main_expr)
        # do_expr requires two changes
        # 1) dag truncation
        # 2) dict substitution
        # For example, if Z(X=1),
        # then dag.truncate('X')
        dag.truncate(','.join([ x[0] for x in do_expr ]))
        # STEP 1 --- Truncate Model if necessary 
        # and remove any variable related to do from main_expr
        # STEP 2 --- Check topological order of truncated DAG
        # just to see who is first in main_var 
        # Get recursively every children
        top_order = dag.get_top_order()
        for v in top_order:
            if v in main_var:
                first_v = v
                break
        funcs = {}
        # Need a recursive function to get parameters from children
        print(f"First v: {first_v}")
        all_children = [first_v ] + list(set(find_vs(first_v, dag)))
        # STEP 3 --- Run find_functions in order
        for v in all_children:
            self.canModel.get_functions(v, main_expr)
        # STEP 4 --- Join parameters according to c-components
#        for v in top_order:
#            if v not in main_var:
#                funcs[v] = self.canModel.iso_params[v]
#            else:
#                funcs[v] = self.canModel.get_values(v, main_expr)
        return ""
        funcs = [ ('', dict([main_expr])) ]
        for v in top_order:
            funcs = get_funcs(self, funcs, v, do_to_dict)
#        for v in top_order:
#            print(v)
#            funcs = [ ( r[0] + ',' + f[0], r[1])  
#                    for f in funcs
#                    for r in self.filter_functions(v, do_to_dict(f[1]))   ]
        funcs = [ [ k for k in x[0].split(',') if k!= '' ] for x in funcs ]
        # STEP 4 --- Separate parameters by c_components
        funcs = [ a for k in funcs for a in get_c_component(k, self.c_parameters) ]
        return funcs

    def parse_irreducible_expr(self, expr):
        """
        It gets an expr such as "Y(X=1, B=0)=1"
        and returns an expresion in terms of canonical model variables
        The procedure is to separate which is the main_var (Y) and its value (1).
        Then, one has to parse all the intervention variables "X=1, B=0".
        It's possible to have empty interventions, for instance, "Y=1"
        Algorithm:
            INPUT: irreducible expression, DAG, and a canModel
        1) Put all variables in descendent topological order and 
        find the first referred in the expression.
         EXAMPLE: A -> B -> Y -> Z, A -> Y, X -> B -> Z, and  (Y=1, B=1).
        Reverse topological order is: Z, Y, B, X, A. It will start with Z. 
        But Z is not contained in expr, so the first will be Y.
        2) For v in V (starting in the first for the reverse topological order),
        OUTPUT: a list of canonical expressions, representing this irreducible expr
        ----------------------------------------------------------------
        Note: There is a difference between irreducible and parse_expr, in general.
      parse_expr will accept condiitonal and joint probability expressions, such 
        as P(Y(X=1)=1, A(B=0)=1 | K(B=1) = 0, A(Y=0) = 1)
        """
        main_expr, do_expr = clean_irreducible_expr(expr) 
        dag = deepcopy(self.dag)
        # do_expr requires two changes
        # 1) dag truncation
        # 2) dict substitution
        # For example, if Z(X=1),
        # then dag.truncate('X')
        # and add2dict
        dag.truncate(','.join([ x[0] for x in do_expr ]))
        do_to_dict = add2dict({x[0]: x[1] for x in do_expr })
        # STEP 1 --- Truncate Model if necessary 
        # STEP 2 --- Check topological order of truncated DAG
        top_order = dag.get_top_order()
        top_order.reverse()
        # STEP 3 --- Run find_functions in order
        funcs = [ ('', dict([main_expr])) ]
        for v in top_order:
            funcs = get_funcs(self, funcs, v, do_to_dict)
#        for v in top_order:
#            print(v)
#            funcs = [ ( r[0] + ',' + f[0], r[1])  
#                    for f in funcs
#                    for r in self.filter_functions(v, do_to_dict(f[1]))   ]
        funcs = [ [ k for k in x[0].split(',') if k!= '' ] for x in funcs ]
        # STEP 4 --- Separate parameters by c_components
        funcs = [ a for k in funcs for a in get_c_component(k, self.c_parameters) ]
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
        exprs = self.collect_worlds(expr)
        exprs = [ self.parse_expr(i,j) for i,j in exprs.items() ]
        return exprs
#        exprs = [ self.parse_irreducible_expr(x.strip()) for x in expr.split('&')]
#        exprs = reduce(lambda a,b: intersect_expr(a,b, self.c_parameters), exprs)
#        exprs = [ tuple(sorted([i for i in x if i != '' ]))  for x in exprs ] # Remove empty ''
#        return sorted(exprs)
    

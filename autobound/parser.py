from autobound.canonicalModel import canonicalModel
from autobound.DAG import DAG
import numpy as np
from itertools import product
from functools import reduce
from copy import deepcopy

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
        do_expr = [ [ x[0], int(x[1]) ] for x in do_expr ]
        main_expr = [ expr.split('(')[0].strip(),
                int(expr.split(')')[1].split('=')[1].strip()) ]
    else:
        main_expr = expr.split('=')
        main_expr = [ main_expr[0], int(main_expr[1]) ]
        do_expr = [ ]
    return (main_expr, do_expr)


def find_involved_c(dag, canModel, main_expr, do_expr):
    """
    STEP 1 in parse_irreducible_expr method.
    For particular dag, canModel, and main_expr and do_expr
    with respect to an irreducible expr,
    find all relevant c_components for the problem
    """
    if len(do_expr) == 0:
        inbetween = [ main_expr[0] ]
    else:
        inbetween = reduce(lambda a,b: a.union(b), 
                [ dag.find_inbetween(x[0], main_expr[0]) 
                for x in do_expr ])
    involved_c = [  c 
            for c in canModel.c_comp 
            if any([x in c for x in inbetween ] ) ]
    return involved_c


def find_possibilities(involved_c, canModel, main_expr, do_expr):
    """
    STEP 2 in parse_irreducible_expr method.
    Find all possibilities in terms of observational values
    for a particular irreducible expr.
    """
    involved_var = [ x for c in involved_c for x in c 
            if x != main_expr[0] and x not in [ k[0] for k in do_expr ] ] 
    number_values = [ list(range(b)) for a,b in canModel.number_values.items() 
            if a in involved_var ]
    possibilities = [ tuple([ y for y in x ] + [ main_expr[1] ] + [ k[1] 
        for k in do_expr ] )  
        for x in list(product(*number_values)) ]
    names = involved_var +  [main_expr[0]] + [ k[0] for k in do_expr ]
    return (names, possibilities)

def test_searchvalue():
    assert (search_value('11010010', '1', (2,2,2)) == np.array([[0,0,0],
                                                               [0,0,1],
                                                               [0,1,1],
                                                               [1,1,0]])).all()
                    

def test_parser():
y = DAG()
y.from_structure("Z -> Y, Z -> X, U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
y.copy()
x = Parser(y)
x.set_dag(y, {'X': 2})

def test_filterfunctions():
    y = DAG()
    y.from_structure("X -> Y, Z -> X, U -> X, U -> Y", unob = "U")
    x = Parser(y, {'X':3})
    assert x.filter_functions('Y', {}) == [ ]
    assert x.filter_functions('Z', {'X': 1}) == [ ]
    assert x.filter_functions('Z', {'Z': 1}) == [('1', {'Z': 1})]
    assert x.filter_functions('Y', {'Y': 1}) == x.filter_functions('Y', {'Y': 1, 'X': 0})
    assert len(x.filter_functions('Y', {'Y': 1})) == 12
    assert len(x.filter_functions('Y', {'Y': 1})) == 4
    y = DAG()
    y.from_structure("A -> Y, B -> Y, U -> B, U -> Y", unob = "U")
    x = Parser(y, {})
    # len(x.filter_functions('Y', {'Y': 1}))
    # len(x.filter_functions('Y', {'Y': 1, 'A': 1}))
    # assert len([ k for k in x.filter_functions('Y', {'A': 1}) if k[0] == '0000' ]) == 0
    # assert len([ k for k in x.filter_functions('Y', {'A': 1}) if k[0] == '0000' ]) == 0
    #x.parse_irreducible_expr('Y(Z=0)=1')
    #x.parse_irreducible_expr('Y=1')


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
    
    def dagval_to_canval(self, var = "", do = ""):
        """ Input: a set of values in terms of dags, for instance, 'X=1,Z=0'
        Output: a set of variables in terms of canonical models
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
                
    def filter_functions(self, v, query):
        """
        Input: a variable v of a particular model 
        and a query with values to be filtered such as {'v': v, 'a': a, ...} 
        Output: a list with correct functions and 
        possible parent values.
        For instance, [('Y111111', {X:'0', Z:'0'} ), 
                       ('Y111111', {X:'0', Z:'1'} )]
        ----------------------------------------
        Algorithm:
        Step 0: If v is not refered in the query, return empty
        Step 1: List all parents of v
        Step 2: Get all canonical variables with respect to v. Call this list functions
        """
        if v not in query.keys():
            return []
        parents = list(self.dag.find_parents_no_u(v))
        parents.sort()
        # Select relevant canonical variables
        functions = list(set([ k for x in self.canModel.parameters for k in x.split('.') ]))
        functions = [ x for x in functions if x.startswith(v) ] 
        functions = [ x.replace(v, '',1) for x in functions if str(query[v]) in x ]
        pa_dimensions = tuple(self.canModel.number_values[k] for k in parents)
        # res_functions is named differently from functions to 
        # emphasize that we are returning not only functions, but also 
        # new queries with respect to parents
        res_functions = [ (x, y)
            for x in functions 
            for y in search_value(x, str(query[v]), pa_dimensions).tolist() ] 
        res_functions = [ ( x[0], dict(zip(parents + [ v ] ,x[1] + [ query[v] ] )) )  # Transforming queries to dict
                for x in res_functions ]  # Has to add  v to parents, and query[v] to x[1]
        # n_relevant refers to the number of variables in the query which is a parent or main
        n_relevant = len([ x for x in parents + [ v ] if x in query.keys() ])
        res_functions = [ ( x[0], x[1] ) 
                for x in res_functions  
                if len(x[1].items() & query.items()) >= n_relevant ] # It has to have more intersections than v: value
        return res_functions
            
    def parse_irreducible_expr(self, expr):
        """
        It gets an expr such as "Y(X=1, B=0)=1"
        and returns an expresion in terms of canonical model variables
        The procedure is to separate which is the main_var (Y) and its value (1).
        Then, one has to parse all the intervention variables "X=1, B=0".
        It's possible to have empty interventions, for instance, "Y=1"
        Algorithm:
            INPUT: irreducible expression, DAG, and a canModel
1) Put all variables in descendent topological order and find the first referred in the expression.
EXAMPLE: A -> B -> Y -> Z, A -> Y, X -> B -> Z, and  (Y=1, B=1).
Reverse topological order is: Z, Y, B, X, A. It will start with Z. But Z is not contained in expr, so the first will be Y.
2) For v in V (starting in the first for the reverse topological order),
Returns all the functions with v = x, for example (Y = 1).
            1) Find all involved c_components: 
                a) contains the main variable;
                b) contains do variables, if not empty;
                c) contains all variables in paths from do to main
            2) List all the possibilities for all variables in terms of 
            values of the DAG
                As an example ,
                consider the frontdoor case with Z -> X -> Y, and Z <---> Y
                For listing all possibilities for Y(Z=1), 
                one has to consider variations of two c-components, 
                {Z,Y}, and {X}, as they are all involved.
                As Y=1 and Z is set to 1, one has to 
                consider a list of DAG possible values, which in this case 
                only refers to X: [(Z=1,X=0,Y=1), (Z=1,X=1,Y=1)]
                possibilities lists all possible tuples, for instance, 
                [ (1,0,1), (1,1,1) ]
                names refers to the column names, for instance,
                [ 'Z', 'X', 'Y' ]
            3) For all values in list, calculate the equivalent canonical expression,
            using method dagval_to_canval
            4) For all found canonical expressions, for all do values, for instance 
            Z = 1, canonical expressions must be generalized over all 
            canonical Zs. For example, if we have canonical (Z=1, X=3, Y = 12), 
            then we have to generalize over (Z=0, X=3, Y = 12), (Z=1,..., 
            and all the other Z values
        OUTPUT: a set of canonical expressions
        ----------------------------------------------------------------
        Note: There is a difference between irreducible and parse_expr, in general.
      parse_expr will accept condiitonal and joint probability expressions, such 
        as P(Y(X=1)=1, A(B=0)=1 | K(B=1) = 0, A(Y=0) = 1)
        """
        main_expr, do_expr = clean_irreducible_expr(expr)
        # STEP 1
        involved_c = find_involved_c(self.dag, self.canModel, main_expr, do_expr)
        # STEP 2
        names, possibilities = find_possibilities(involved_c, self.canModel, main_expr, do_expr)
        # STEP 3
        top_order = self.dag.get_top_order()
    


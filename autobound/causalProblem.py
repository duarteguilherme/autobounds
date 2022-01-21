from .canonicalModel import canonicalModel
from .DAG import DAG
from .Parser import Parser
#from autobound.canonicalModel import canonicalModel
#from autobound.DAG import DAG
#from autobound.Parser import Parser
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from functools import reduce

class Program:
    """ This class
    will state a optimization program
    to be translated later to any 
    language of choice, pyscipopt (pip, pyscipopt(cip),
    pyomo, among others
    A program requires first parameters, 
    an objective function,
    and constraints
    """
    def __init__(self):
        self.parameters = [ ]
        self.estimand = tuple()
        self.constraints = [ tuple() ]
    
    def to_pyomo(self):
        pass
    
    def to_pip(self):
        pass
    
    def to_cip(self):
        pass



def test_add_constraints():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y, K -> X", unob = "U")
    x = causalProblem(y, {'X': 2})
    assert (1, 'Z0') in x.parameters
    x.set_p_to_zero(['Z0'])
    assert (0, 'Z0') in x.parameters
    x.add_constraint([(-0.15, ['1']), (-0.15, ['1']), (1, ['X1111']), (-1, ['X1111', 'Z1']), (2, ['X1111'])])
    x.constraints
    assert [(-0.3, ['1']), (3, ['X1111']), (-1, ['X1111', 'Z1'])] in x.constraints
    x.add_constraint([(1, ['X1110']), (-1, ['X1110', 'Z1']), (-1, ['X1110'])])
    assert [(-1, ['X1110', 'Z1'])] in x.constraints

def test_check_constraints():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.load_data(datafile)
    x.add_prob_constraints()
    x.check_constraints()
    assert (0.5, ['X00.Y00', '1']) in x.constraints[0]


def test_causalproblem():
y = DAG()
y.from_structure("Z -> X, X -> Y, U -> X, U -> Y, K -> X", unob = "U")
x = causalProblem(y, {'X': 2})
x.write_program()

x.constraints[1]
def test_replace_first_nodes():
    assert replace_first_nodes([('Z0', 0.5), ('Z1', 0.5)], 
            (1, ['X00.Y10', 'Z0'])) == (0.5, ['X00.Y10', '1'])


def replace_first_nodes(first_nodes, constraint):
    """ 
    Gets an expr inside a constraint, for instance,
    (1, ['X00.Y00', 'Z0']) and if Z0 is in first nodes, 
    it replace Z0 by 1 and it multiplies 1 by the prob
    """
    coef, var = constraint[0], constraint[1]
    for i,v in enumerate(var):
        for n in first_nodes:
            if v == n[0]:
                var[i] = '1'
                coef *= n[1]
                break
    return ( coef, var )
            




def get_constraint_from_row(row_data, row_prob, parser):
    """ 
    Function to be employed in load_data method in causalProgram
    One introduces the row data , row prob , Parser
    and function returns constraints 
    """
    query = [ f'{row_data.index[j]}={int(row_data[j])}'
                    for j,k in enumerate(list(row_data)) ]
    return [( -1 * row_prob, [ '1' ])] + [ (1, [ i for i in x ]) 
            for x in parser.parse('&'.join(query)) ]


class causalProblem:
    def __init__(self, dag, number_values = {}):
        """
        Causal problem has to have three elements: 
            a) canonicalModel: a canonical model;
            b) estimand to be optimized over;
            c.1) data;
            c.2) other constraints;
        sense is not mandatory anymore, but a function optimize that users can choose sense and optimizer.
        A parser needs to be included to translate expressions to canonicalModel language
        unconf_roots corresponds to roots for which we have data 
        """
        self.canModel = canonicalModel()
        self.dag = dag
        self.canModel.from_dag(self.dag, number_values)
        self.Parser = Parser(dag, number_values)
        self.parameters = [ (1, x) for x in self.canModel.parameters ]
        self.estimand = [ ]
        self.constraints = [ ]
        self.unconf_first_nodes = [ ]
        
    def write_program(self):
        self.program = Program()
        unconf_roots = self.find_unconf_roots()
        self.program.parameters = [ x[1] 
                for x in self.parameters 
                if x[0] == 1 or x[1] not in [ i[0] for i in self.unconf_roots ] ]
        # Add default constraints
        # Add probabilistic contraints
#        for c in self.Parser.c_parameters:
#            if 
        
    def check_constraints(self):
        """ 
        Check all constraints 
        and replace values for unconf_first_nodes
        """
        self.constraints = [ [ replace_first_nodes(self.unconf_first_nodes, y) 
            for y in x ]  
                for x in self.constraints ] 
    
    def add_prob_constraints(self):
        """
        """
        unconf_nodes = [ x[0] for x in self.unconf_first_nodes ] 
        not_0_parameters = [ x[1] for x in self.parameters if x[0] != 0 ]
        for c in self.Parser.c_parameters:
            prob_constraints = [ (1, [ x ]) 
                        for x in c
                if x in not_0_parameters 
                and x not in unconf_nodes ]
            if len(prob_constraints) > 0:
                self.add_constraint(prob_constraints)
    
    def load_data(self, filename, cond = False):
        """ It accepts a file 
        file must be csv. Columns will be added if they match parameters...
        Column prob must indicate probability.
        For example,
        >    X,Y,prob,
        >    1,0,0.25,
        >    0,1,0.25,
        >    1,1,0.25,
        >    0,0,0.25
        Algorithm: 
        1) If data is not conditioned, method must fill out unconf_first_nodes info
        2) Data must be added as constraint
        """
        datam = pd.read_csv(filename) 
        columns = [ x for x in datam.columns if x in list(self.dag.V) ]  + ['prob']
        datam = datam[columns]
        first_nodes = [ k for k in self.dag.find_first_nodes() 
                if len(self.dag.find_u_linked(k)) == 0 and k in columns]
        if not cond:
            for k in first_nodes:
                self.unconf_first_nodes += [ (k + str(i), 
                    datam.groupby(k).sum().loc[i]['prob'] )
                        for i in datam.groupby(k).sum().index ]
        self.set_p_to_zero([ x[0] for x in self.unconf_first_nodes ])
        # Adding data
        column_rest = [x for x in columns if x not in first_nodes and x!= 'prob']
        grouped_data = datam.groupby(column_rest).sum()['prob'].reset_index()
        for i, row in grouped_data.iterrows():
            self.add_constraint(get_constraint_from_row(row[column_rest], 
                    row['prob'], self.Parser))
    
    def set_p_to_zero(self, parameter_list):
        """
        For a particular list  of parameters
        ['X0111', 'Z0'], set them to 0
        """
        self.parameters = [ (x[0], x[1])
                for x in self.parameters  
                if x[1] not in parameter_list ] + [ (0, x) 
                        for x in parameter_list ]
    
    def add_constraint(self, constraint):
        """
        Input: list of tuples with constant and 
        statemenets. For example [(-1, ['X1111', 'Z1']), (2, ['X1111'])]
        """
        # Sorting constraint
        constraint = [ (x[0], sorted(x[1])) for x in constraint ] 
        constraint = [ [x[0], sorted(x[1])] for x in constraint ] 
        expr_list = [ x[1]  # Removing duplicated
                for n, x in enumerate(constraint) 
                if x[1] not in [ i[1] for i in constraint[:n] ] ]
        constraint = [ (sum([ i[0] 
            for i in constraint if x == i[1] ]), x)
                for x in expr_list ]
        constraint = [ (x[0], x[1]) for x in constraint if x[0] != 0 ]
        self.constraints.append(constraint)
    
    def set_estimand(self,estimand, cond = ['%']):
        """
        Input: an expression similar to a constraint
        """
        self.add_constraint(estimand + 
                [ (-1, ['objvar', x ]) for x in cond ]  )
    
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
        for i in [0,1]:
            cons1.append(quicksum([ self.parameters[k]
                    for k in 
                    self.get_response_expr({keys[i]: var[keys[i]]}) ] ))
        cons2 = quicksum([ self.parameters[k]
                for k in 
                self.get_response_expr(var) ])
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
    






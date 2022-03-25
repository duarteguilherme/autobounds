from .canonicalModel import canonicalModel
from .Program import Program
from .DAG import DAG
from .Parser import Parser
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from functools import reduce

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
            


def transform_constraint(constraint):
    """ 
    To be used inside write_program method
    This functions gets a constraint in 
    causalProblem format and translate it to 
    program format
    """
    res =  [ ['' if k[0] == 1 else str(k[0]) ] + 
            [ i for i in k[1] if i != '1' ] 
        for k in constraint ] 
    res = [ [ j for j in i if j != '' ] for i in res  ]
    return res



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
        
    def query(self, expr, sign = 1):
        """ 
        Important function:
        This function is exactly like parse in Parser class.
        However, here it returns a constraint structure.
        So one can do causalProgram.query('Y(X=1)=1') in order 
        to get P(Y(X=1)=1) constraint.
        sign can be 1, if positive, or -1, if negative.
        """
        return [ (sign, list(x)) for x in self.Parser.parse(expr) ]
    
    def write_program(self):
        """ It returns an object Program
        """
        program = Program()
        self.check_constraints()
        program.parameters = [ x[1] 
                for x in self.parameters 
                if x[0] == 1 ] + [ 'objvar']
        program.constraints = [
                transform_constraint(x )
                for x in self.constraints
                ]
        return program
        
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
                prob_constraints += [ (-1.0, ['1'])]
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
        # First nodes ---- parameters have to be set to 0
        # This part might be refactored in a different method
        if not cond:
            for k in first_nodes:
                self.unconf_first_nodes += [ (k + str(i), 
                    datam.groupby(k).sum().loc[i]['prob'] )
                        for i in datam.groupby(k).sum().index ]
        self.set_p_to_zero([ x[0] for x in self.unconf_first_nodes ])
        # Adding data
#        column_rest = [x for x in columns if x not in first_nodes and x!= 'prob']
        column_rest = [x for x in columns if x!= 'prob']
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
    
    def set_estimand(self,estimand, cond = ['1']):
        """
        Input: an expression similar to a constraint
        This algorithm there will 
        add estimand as a constraint with a new variable 
        objvar that will be added as a parameter.
        If the estimand is conditioned, then this condition 
        is multiplied by objvar, according to the algebraic formula.
        P(Y|X) = P(Y,X)/P(X) = objvar, then P(Y,X) - P(X) * objvar = 0
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
    






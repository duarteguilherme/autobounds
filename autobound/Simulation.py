from .canonicalModel import canonicalModel
from .DAG import DAG
from .Parser import Parser
from autobound.canonicalModel import canonicalModel
from autobound.DAG import DAG
from autobound.Parser import Parser
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from functools import reduce



def test_simulation():
dag = DAG()
dag.from_structure('X -> Y, Z -> X, U -> Y, U -> X')
x = Simulation()


class Simulation:
    def __init__(self, dag, number_values = {}):
        """
        Simulation object has to have only one element. 
            a) canonicalModel: a canonical model;
            b) parameters: those are parameters of 
            the canonical model. It will be represented 
            as a dictionary and values.
        It is optional if one wants to restrict the values of 
        some canonical parameters to some numerical value. 
        This is relevant when one wants to set the values for the ATE or 
        other estimand.
        """
        self.canModel = canonicalModel()
        self.dag = dag
        self.canModel.from_dag(self.dag, number_values)
        self.Parser = Parser(dag, number_values)
        self.parameters = [ (1, x) for x in self.canModel.parameters ]
        
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
    
    def set_param_value(self, param_list):
        if param_list is not list:
            param_list = [ param_list ] 
        for i in param_list:
            if i is not dict:
                raise Exception("Introduce a dictionary or a list of dictionaries. Format: {'R00':0.04}")

    def convert
    def simulate(self):
        """  For all the parameters not yet determined,
        it derives values for them utilizing uniform and 
        dirichlet distributions
        """
        pass



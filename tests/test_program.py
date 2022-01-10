#from autobound.autobound.program import program
#import numpy.random as rd
#import pytest
#import os
#
#
#def test_fromdag():
#    x = causalProgram()
#    x.from_dag(
#            DAG().from_structure("U -> X, X -> Y, U -> Y, Uy -> Y", unob = "U , Uy")
#            )
#    assert x.V == set(('Y', 'X'))
#    assert x.E == set((('Uy', 'Y'), ('X','Y'),('U','Y'),('U','X')))
#    assert x.U == set(('Uy','U'))


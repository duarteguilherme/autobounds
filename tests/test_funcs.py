from autobound.autobound.funcs import *
import numpy.random as rd
import pytest
import os


def test_create_function():
    parents = [ "U_XY", "U_X", "V_Z"]
    rd.seed(500)
    test = create_function(parents)
    x = test[0]((0,0,1))
    assert x == 1
    rd.seed(501)
    test = create_function(parents)
    x = test[0]((0,0,1))
    assert x == 1

from numpy import random

funcs = [
        lambda a,b: a and b,
        lambda a,b: a or b,
        lambda a,b: int((a and not b) or (b and not a)),
        lambda a,b: int((a and not b) or (b and not a))
        ]


def pick_func():
    n = random.choice(4,1)[0]
    return (n, funcs[n])


def create_function(parents):
    """ For given parents, it assumes everything is binary
    and creates a function from parents to a variable 
    based on dyadic relationship between them (only AND, OR, and XOR).
    Input is a tuple of variables (strings).
    """
    if len(parents) < 1:
        raise Exception("Error. Number of parents is less than zero")
    funs = [(9,lambda a: a)] + [ pick_func() for x in range(1,len(parents)) ]
    def anon(tupl):
        if len(tupl) != len(funs):
            raise Exception("Length of tuple does not match the number of functions")
        res = funs[0][1](tupl[0])
        for i in range(1,len(funs)):
            res = funs[i][1](res, tupl[i]) 
        return res
    return (anon, [x[0] for x in funs])


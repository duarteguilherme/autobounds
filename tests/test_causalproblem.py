from autobound.autobound.DAG import DAG
from autobound.autobound.causalProblem import causalProblem
import io
from copy import deepcopy

def test_load_data():
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.125
    0,0,1,0.125
    0,1,0,0.125
    0,1,1,0.125
    1,0,0,0.125
    1,0,1,0.125
    1,1,0,0.125
    1,1,1,0.125''')
    datafile2 = deepcopy(datafile)
    y = DAG()
    y.from_structure("Z -> Y, X -> Y, U -> X, U -> Y", unob = 'U')
    x = causalProblem(y, {'X': 2})
    x.load_data(datafile)
    x.add_prob_constraints()
    x.constraints[3] == [(-1, ['0.25']), (1, ['X1.Y0001', 'Z1']), 
            (1, ['X1.Y0010', 'Z0']), (1, ['X1.Y0011', 'Z0']), 
            (1, ['X1.Y0011', 'Z1']), (1, ['X1.Y0101', 'Z1']), 
            (1, ['X1.Y0110', 'Z0']), (1, ['X1.Y0111', 'Z0']), 
            (1, ['X1.Y0111', 'Z1']), (1, ['X1.Y1001', 'Z1']), 
            (1, ['X1.Y1010', 'Z0']), (1, ['X1.Y1011', 'Z0']), 
            (1, ['X1.Y1011', 'Z1']), (1, ['X1.Y1101', 'Z1']), 
            (1, ['X1.Y1110', 'Z0']), (1, ['X1.Y1111', 'Z0']), (1, ['X1.Y1111', 'Z1'])] 
    y = DAG()
    y.from_structure("Z -> Y, U -> Z, X -> Y, U -> Y, U -> X", unob = "U")
    x = causalProblem(y, {'X': 2})
    x.load_data(datafile2)
    x.add_prob_constraints()
    x.constraints[1] == [(-1, ['0.125']), (1, ['X0.Y0000.Z1']), 
            (1, ['X0.Y0001.Z1']), (1, ['X0.Y0010.Z1']), (1, ['X0.Y0011.Z1']), 
            (1, ['X0.Y1000.Z1']), (1, ['X0.Y1001.Z1']), (1, ['X0.Y1010.Z1']), (1, ['X0.Y1011.Z1'])] 



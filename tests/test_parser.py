from autobound.autobound.DAG import DAG
from autobound.autobound.Parser import *



def test_searchvalue():
    assert (search_value('11010010', '1', (2,2,2)) == np.array([[0,0,0],
                                                               [0,0,1],
                                                               [0,1,1],
                                                               [1,1,0]])).all()
                    

def test_filterfunctions():
    y = DAG()
    y.from_structure("X -> Y, Z -> X, U -> X, U -> Y", unob = "U")
    x = Parser(y, {'X':3})
    assert x.filter_functions('Y', {}) == [('', {})]
    assert x.filter_functions('Z', {'X': 1}) == [('', {'X': 1})]
    assert x.filter_functions('Z', {'Z': 1}) == [('Z1', {'Z': 1})]
    assert len(x.filter_functions('Y', {'Y': 1, 'X': 0})) < len(x.filter_functions('Y', {'Y': 1}))
    y = DAG()
    y.from_structure("X -> Y, Z -> Y, U -> Z, U -> Y", unob = "U")
    x = Parser(y, {})
    assert len([ k for k in x.filter_functions('Y', {'A': 1}) if k[0] == '0000' ]) == 0
    assert len([ k for k in x.filter_functions('Y', {'A': 1}) if k[0] == '0000' ]) == 0

def test_parse():
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    assert x.parse('Y=1&X=0') == [('X00.Y10', 'Z0'), ('X00.Y10', 'Z1'), 
            ('X00.Y11', 'Z0'), ('X00.Y11', 'Z1'), ('X01.Y10', 'Z0'), 
            ('X01.Y11', 'Z0'), ('X10.Y10', 'Z1'), ('X10.Y11', 'Z1')]
    assert x.parse('Y = 1& Y = 0') == [] 
    assert x.parse('Y(X=1)=1& Y(X=0)=1') == [('X00.Y11',), ('X01.Y11',), ('X10.Y11',), ('X11.Y11',)]
    y = DAG()
    y.from_structure("Z -> Y, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    x.parse('Y=1 &X=1')
    assert x.parse('Y=1 &X=1') == [('X1.Y0001', 'Z1'), ('X1.Y0010', 'Z0'), ('X1.Y0011', 'Z0'), 
            ('X1.Y0011', 'Z1'), ('X1.Y0101', 'Z1'), ('X1.Y0110', 'Z0'), ('X1.Y0111', 'Z0'), 
            ('X1.Y0111', 'Z1'), ('X1.Y1001', 'Z1'), ('X1.Y1010', 'Z0'), ('X1.Y1011', 'Z0'),
            ('X1.Y1011', 'Z1'), ('X1.Y1101', 'Z1'), ('X1.Y1110', 'Z0'), ('X1.Y1111', 'Z0'), ('X1.Y1111', 'Z1')] 

def test_parse_irreducible():
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    part1 = [('X00.Y11',), ('X01.Y11',), ('X10.Y11',), ('X11.Y11',), ('X00.Y10',), 
            ('X01.Y10',), ('X10.Y10',), ('X11.Y10',)]
    assert set(part1) == set(x.parse_irreducible_expr('Y (X =0) = 1'))
    part2 = [('X11.Y01',), ('X10.Y01',), ('X01.Y11',), ('X00.Y11',), ('X11.Y11',), 
            ('X10.Y11',), ('X01.Y10',), ('X00.Y10',)] 
    assert set(x.parse_irreducible_expr('Y (Z =0)= 1')) == set(part2)


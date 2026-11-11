from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Parser import *
from autobounds.autobounds.Q import compare_lists
from copy import deepcopy
from itertools import product
import pytest


@pytest.mark.parametrize('structure,latent,cards', [
    ('Z -> X, X -> Y, U -> X, U -> Y', 'U', {'Z': 2, 'X': 2, 'Y': 2}),
    ('Z -> X, X -> Y, U -> X, U -> Y', 'U', {'Z': 3, 'X': 3, 'Y': 3}),
    ('Z -> X, X -> Y', '', {'Z': 2, 'X': 2, 'Y': 2}),
    ('Z -> Y, X -> Y, U -> X, U -> Z', 'U', {'Z': 2, 'X': 2, 'Y': 2}),
])
def test_indexed_queries_match_response_scans(monkeypatch, structure, latent, cards):
    parser = Parser(DAG(structure, unob=latent), dict(cards))
    events = [
        '&'.join(f'{v}={value}' for v, value in zip(('Z', 'X', 'Y'), row))
        for row in product(*(range(cards[v]) for v in ('Z', 'X', 'Y')))
    ]
    events += ['Y=1', 'X=0', 'Y(X=0)=1', 'Y(X=1)=1',
               'Y(X=0)=1&Y(X=1)=1', 'Y=1&X(Z=0)=0',
               'Y(X=1,Z=1)=1', 'Y=0&Y=1']
    original_index = deepcopy(parser._response_index)
    indexed = [parser.parse(event) for event in events]
    with monkeypatch.context() as patch:
        patch.setattr(parser, '_get_c_component',
                      lambda functions: get_c_component(functions, parser.c_parameters))
        scanned = [parser.parse(event) for event in events]
    assert indexed == scanned
    assert parser._response_index == original_index
    copied = deepcopy(parser)
    assert [copied.parse(event) for event in events] == indexed


def test_index_preserves_components_and_matches_whole_tokens():
    parser = Parser(DAG('X -> Y, U -> X, U -> Y, XX -> W', unob='U'))
    assert parser._get_c_component(['X0']) == [
        ('X0.Y00',), ('X0.Y01',), ('X0.Y10',), ('X0.Y11',)
    ]
    assert parser._get_c_component(['X0', 'X1']) == []
    matches = parser._get_c_component(['X0', 'XX1'])
    assert len(matches) == 4
    assert all(set(row) == {'X0.' + y, 'XX1'}
               for row, y in zip(matches, ['Y00', 'Y01', 'Y10', 'Y11']))


def test_indexed_program_preserves_constraints_and_scip_bounds(monkeypatch):
    import pandas as pd
    from autobounds.autobounds import Bounder

    def build_program():
        problem = Bounder(DAG('Z -> X, X -> Y, U -> X, U -> Y', unob='U'))
        problem.set_ate('X', 'Y')
        data = pd.DataFrame(list(product(range(2), repeat=3)), columns=['Z', 'X', 'Y'])
        data['prob'] = 0.125
        problem.load_data(data)
        problem.add_prob_constraints()
        return problem.write_program()

    indexed = build_program()
    with monkeypatch.context() as patch:
        patch.setattr(Parser, '_get_c_component',
                      lambda self, functions: get_c_component(functions, self.c_parameters))
        scanned = build_program()
    assert indexed.parameters == scanned.parameters
    assert indexed.constraints == scanned.constraints
    for program in (indexed, scanned):
        lower, upper, _, _ = program.run_scip(verbose=False, maxtime=5)
        assert lower['primal'] == pytest.approx(-0.5)
        assert lower['dual'] == pytest.approx(-0.5)
        assert upper['primal'] == pytest.approx(0.5)
        assert upper['dual'] == pytest.approx(0.5)

def test_parse_irreducible():
    # Testing parse_expr
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U")
    x = Parser(y, {'X': 2})
    x.parse_expr('X=0', ['Y=1'])
    part1 = [('X00.Y11',), ('X01.Y11',), ('X10.Y11',), ('X11.Y11',), ('X00.Y10',), 
            ('X01.Y10',), ('X10.Y10',), ('X11.Y10',)]
    assert set(part1) == set(x.parse_expr('X=0', ['Y=1']))
    part2 = [('X11.Y01',), ('X10.Y01',), ('X01.Y11',), ('X00.Y11',), ('X11.Y11',), 
            ('X10.Y11',), ('X01.Y10',), ('X00.Y10',)] 
    assert set(x.parse_expr('Z=0',['Y=1'])) == set(part2)


def test_edge_active():
    d0 = Parser(DAG('D -> Y'), {'Y':3})
    d1 = Parser(DAG('D -> Y, A -> Y, B -> Y'))
    d2 = Parser(DAG('D -> M, A -> Y, B -> Y'))
    assert (len(d2.is_active('D -> Y')) == 0)
#    print(d0.is_active('D -> Y'))
    res2 = set(['Y0010', 'Y0011', 'Y0110', 'Y0111', 'Y0001', 'Y1001', 'Y1011', 'Y1000', 'Y1100', 'Y1101', 'Y0100', 'Y1110'])
    assert (set(d2.is_active('A -> Y')) == res2)



def test_parse_complete():
    dag = DAG()
#    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    dag.from_structure("Z -> X, X -> Y")
    parser = Parser(dag)
    print(parser.parse('Y=1&X(Z=0)=0'))
    print(parser.parse('Y=1&X(Z=0)=0', complete = True))
    print(parser.parse('X(Z=0)=0', complete = True))
    print(parser.parse('Y=1', complete = True))




def test_factor_by_ccomponent():
    dag = DAG()
    dag.from_structure("V -> Z, Z -> X, U -> X", unob = "U")
    test_p = Parser(dag)
    ancestors = list(dag.find_ancestors(['V','Z'], no_v = False))
    ancestors = [ i for i in dag.get_top_order() if i in ancestors ] # Put them in order
    all_paths = test_p.translate({'V': 1, 'Z': 0 }, [], ancestors)
    c_comp_base = [  set([j for i in c for j in i.split('.') ]) 
        for c in test_p.c_parameters]
    res_factor = factor_by_c_component(all_paths[0][1], c_comp_base, test_p.c_parameters) 
    assert res_factor[0] == ['V1']
    assert res_factor[1][1] == 'Z10'
    assert res_factor[2][3] == 'X11'


def test_collect_worlds():
    dag = DAG()
    dag.from_structure("V -> Z, V -> X, Z -> X, Z -> W, Z -> Y, W -> Y, X -> Y, U -> X, U -> Y", unob = "U")
    test_p = Parser(dag)
    assert test_p.collect_worlds("Y=0") == {'': ['Y=0']}
    assert test_p.collect_worlds("Y(X=1)=0") == {'X=1': ['Y=0']}
    assert test_p.collect_worlds("Y=0&X=0&X(Z=1)=0&V(Z=0,X=0)=0&W(Z=0,X=0)=1&Y(Z=0,X=0)=0") == {'': ['Y=0', 'X=0'], 'Z=1': ['X=0'], 'Z=0,X=0': ['V=0', 'W=1', 'Y=0']}

def test_searchvalue():
    assert (search_value('11010010', '1', (2,2,2)) == np.array([[0,0,0],
                                                               [0,0,1],
                                                               [0,1,1],
                                                               [1,1,0]])).all()
                    

def test_parse_proxy_graph():
    dag = DAG()
    dag.from_structure("W -> X, W -> Y, W -> P, X -> Y", unob = "U")
    parser = Parser(dag)
    assert set(parser.parse('P=0&Y=0&X=0')) == set([('P00', 'W0', 'X00', 'Y0000'), ('P00', 'W0', 'X00', 'Y0001'), ('P00', 'W0', 'X00', 'Y0010'), ('P00', 'W0', 'X00', 'Y0011'), ('P00', 'W0', 'X00', 'Y0100'), ('P00', 'W0', 'X00', 'Y0101'), ('P00', 'W0', 'X00','Y0110'), ('P00', 'W0', 'X00', 'Y0111'), ('P00', 'W0', 'X01', 'Y0000'), ('P00', 'W0', 'X01', 'Y0001'), ('P00', 'W0', 'X01', 'Y0010'), ('P00', 'W0', 'X01', 'Y0011'), ('P00', 'W0', 'X01', 'Y0100'), ('P00', 'W0', 'X01', 'Y0101'), ('P00', 'W0', 'X01', 'Y0110'), ('P00', 'W0', 'X01', 'Y0111'), ('P00', 'W1', 'X00', 'Y0000'), ('P00', 'W1', 'X00', 'Y0001'), ('P00', 'W1', 'X00', 'Y0100'), ('P00', 'W1', 'X00', 'Y0101'), ('P00', 'W1', 'X00', 'Y1000'), ('P00', 'W1', 'X00', 'Y1001'), ('P00', 'W1', 'X00', 'Y1100'), ('P00', 'W1','X00', 'Y1101'), ('P00', 'W1', 'X10', 'Y0000'), ('P00', 'W1', 'X10', 'Y0001'), ('P00', 'W1', 'X10', 'Y0100'), ('P00', 'W1', 'X10', 'Y0101'), ('P00', 'W1', 'X10', 'Y1000'), ('P00', 'W1', 'X10', 'Y1001'), ('P00', 'W1', 'X10', 'Y1100'), ('P00', 'W1', 'X10', 'Y1101'), ('P01', 'W0', 'X00', 'Y0000'), ('P01', 'W0', 'X00', 'Y0001'), ('P01', 'W0', 'X00', 'Y0010'), ('P01', 'W0', 'X00', 'Y0011'), ('P01', 'W0', 'X00', 'Y0100'), ('P01', 'W0', 'X00', 'Y0101'), ('P01', 'W0', 'X00', 'Y0110'), ('P01', 'W0', 'X00', 'Y0111'), ('P01', 'W0', 'X01', 'Y0000'), ('P01', 'W0', 'X01', 'Y0001'), ('P01', 'W0', 'X01', 'Y0010'), ('P01', 'W0', 'X01', 'Y0011'), ('P01', 'W0', 'X01', 'Y0100'), ('P01', 'W0', 'X01', 'Y0101'), ('P01', 'W0', 'X01', 'Y0110'), ('P01', 'W0', 'X01', 'Y0111'), ('P10', 'W1', 'X00', 'Y0000'), ('P10', 'W1', 'X00', 'Y0001'), ('P10', 'W1', 'X00', 'Y0100'), ('P10', 'W1', 'X00', 'Y0101'), ('P10', 'W1', 'X00', 'Y1000'), ('P10', 'W1', 'X00', 'Y1001'), ('P10', 'W1', 'X00', 'Y1100'), ('P10', 'W1', 'X00', 'Y1101'), ('P10', 'W1', 'X10', 'Y0000'),('P10', 'W1', 'X10', 'Y0001'), ('P10', 'W1', 'X10', 'Y0100'), ('P10', 'W1', 'X10', 'Y0101'), ('P10', 'W1', 'X10', 'Y1000'), ('P10', 'W1', 'X10', 'Y1001'), ('P10', 'W1', 'X10', 'Y1100'), ('P10', 'W1', 'X10', 'Y1101')])

def test_parse_iv_graph():
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    x.parse('Y(X=0)=1')
    assert set(x.parse('Y=1&X=0')) == set([('X00.Y10', 'Z0'), ('X00.Y10', 'Z1'), 
            ('X00.Y11', 'Z0'), ('X00.Y11', 'Z1'), ('X01.Y10', 'Z0'), 
            ('X01.Y11', 'Z0'), ('X10.Y10', 'Z1'), ('X10.Y11', 'Z1')])
    assert x.parse('Y = 1& Y = 0') == [] 
    assert set(x.parse('Y(X=1)=1& Y(X=0)=1')) == set([('X00.Y11',), ('X01.Y11',), ('X10.Y11',), ('X11.Y11',)])
    y = DAG()
    y.from_structure("Z -> Y, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    assert set(x.parse('Y(X=1,Z=1)=1')) == set([('X0.Y0001',), ('X0.Y0011',), ('X0.Y0101',), ('X0.Y0111',), ('X0.Y1001',), ('X0.Y1011',), ('X0.Y1101',), ('X0.Y1111',), ('X1.Y0001',), ('X1.Y0011',), ('X1.Y0101',), ('X1.Y0111',), ('X1.Y1001',), ('X1.Y1011',), ('X1.Y1101',), ('X1.Y1111',)])
    assert x.parse('Y=1 &X=1') == [('X1.Y0001', 'Z1'), ('X1.Y0010', 'Z0'), ('X1.Y0011', 'Z0'), 
            ('X1.Y0011', 'Z1'), ('X1.Y0101', 'Z1'), ('X1.Y0110', 'Z0'), ('X1.Y0111', 'Z0'), 
            ('X1.Y0111', 'Z1'), ('X1.Y1001', 'Z1'), ('X1.Y1010', 'Z0'), ('X1.Y1011', 'Z0'),
            ('X1.Y1011', 'Z1'), ('X1.Y1101', 'Z1'), ('X1.Y1110', 'Z0'), ('X1.Y1111', 'Z0'), ('X1.Y1111', 'Z1')] 



def test_translate():
    # Testing translate
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U")
    x = Parser(y, {'X': 2})
    main_expr1 = {'Y': 1}
    do_expr1 = [['X', 1]]
    ancestors1 = ['Z','Y']
    translation = x.translate(main_expr = main_expr1, do_expr = do_expr1, ancestors = ancestors1)
    assert 'Y01' in translation[1][1][1]
    assert 'Z0' in translation[0][1][0]
    assert 'Y11' in translation[1][1][1]


def test_parse_cond():
    d0 = Parser(DAG('D -> Y, U -> D, U -> Y', unob = 'U'))
    py0_d1 = d0.p('Y=0', 'D=1')
    assert compare_lists(py0_d1._event,  [(1, ['D1.Y00']), (1, ['D1.Y10'])])
    assert compare_lists(py0_d1._cond,  [(1, ['D1.Y00']), (1, ['D1.Y01']), (1, ['D1.Y10']), (1, ['D1.Y11'])])


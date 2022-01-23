#from autobound.causalProblem import causalProblem
#from autobound.DAG import DAG
#import io 

def test_pip_join_expr():
    assert pip_join_expr(['0.5', 'X00.Y00'], ['X00.Y00', 'Z1', 'Z0']) == '0.5 X00.Y00'
    assert pip_join_expr(['0.5'], ['X00.Y00', 'Z1', 'Z0']) == '0.5'
    assert pip_join_expr(['X00.Y00'], ['X00.Y00', 'Z1', 'Z0']) == 'X00.Y00'
    assert pip_join_expr(['0.5', 'X00.Y00', 'Z1'], ['X00.Y00', 'Z1', 'Z0']) == '0.5 X00.Y00 * Z1'

def pip_join_expr(expr, params):
    """ 
    It gets an expr and if there is a coefficient, it 
    separates without using * .
    It is required as a simple list join is insufficient 
    to put program in pip format
    """
    coef = ''.join([x for x in expr if x not in params ])
    expr_rest = ' * '.join([ x for x in expr if x in params ])
    coef = coef + ' ' if coef != '' and expr_rest != '' else coef 
    return coef + expr_rest

def test_program_iv():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.constraints
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    z.to_pip('/home/beta/test_iv.pip')

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
        self.constraints = [ tuple() ]
    
    def to_pyomo(self):
        pass
    
    def to_pip(self, filename, sense = 'max'):
        filep = open(filename, 'w')
        sense = 'MAXIMIZE' if sense == 'max' else 'MINIMIZE'
        filep.write(sense + '\n' + '  obj: objvar' + '\n')
        filep.write('\nSUBJECT TO\n')
        for i, c in enumerate(self.constraints):
            filep.write(f'  a{i}: ' + ' + '.join([ 
               pip_join_expr(k, self.parameters) for k in c ]) 
                + ' = 0\n') 
        filep.write('\nBOUNDS\n')
        for p in self.parameters:
            if p != 'objvar':
                filep.write(f'  0 <= {p} <= 1\n')
            else:
                filep.write(f'  -1 <= {p} <= 1\n')
        filep.write('\nEND')
        filep.close()
    
    def to_cip(self):
        pass


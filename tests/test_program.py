from autobound.autobound.causalProblem import causalProblem
from autobound.autobound.DAG import DAG
import io

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
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    b = z.run_pyomo()
    assert b[0] <= -0.48
    assert b[0] >= -0.52
    assert b[1] <= 0.52
    assert b[1] >= 0.48
#    z.to_pip('/home/beta/test_iv.pip')

from autobounds.autobounds.causalProblem import causalProblem
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Program import change_constraint_parameter_value
from autobounds.autobounds.ProgramUtils import is_linear
from autobounds.autobounds.Query import Query
import io
import time
import pandas as pd


def test_is_linear():
    df = pd.DataFrame(
            {'Z': [0,0,0,0,1,1,1,1],
             'W': [0,0,1,1,0,0,1,1],
             'Y': [0,1,0,1,0,1,0,1],
          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
             })
    dag = DAG()
    dag.from_structure('U -> Z, Z -> W, U -> Y, W -> Y')
    pro = causalProblem(dag)
    pro.load_data(df)
    pro.set_ate('W','Y')
    pro.add_prob_constraints()
    program = pro.write_program()
    assert is_linear(program.constraints[-1])

#def test_separation_lin_nonlin():
#    df = pd.DataFrame(
#            {'Z': [0,0,0,0,1,1,1,1],
#             'W': [0,0,1,1,0,0,1,1],
#             'Y': [0,1,0,1,0,1,0,1],
#          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
#             })
#    dag = DAG()
#    dag.from_structure('U -> Z, Z -> W, U -> Y, W -> Y')
#    pro = causalProblem(dag)
#    pro.add_constraint(pro.query('Z(U=0)=0') - Query(0.43))
#    pro.add_constraint(pro.query('Z(U=0)=1') - Query(0.25))
#    pro.add_constraint(pro.query('Z(U=1)=0') - Query(0.35))
#    pro.add_constraint(pro.query('Z(U=1)=1') - Query(0.15))
#    pro.load_data(df)
#    pro.set_ate('W','Y')
#    pro.add_prob_constraints()
#    program = pro.write_program()
#    constraints1 = program.constraints
#    program.simplify_linear()
#    constraints2 = program.constraints
#    print(len(constraints1))
##    assert is_linear(program.constraints[-1])
#

def test_rdd_paper():
    # P(U=1) = 0.4
    p_u = pd.DataFrame({
        'U': [0, 1],
        'prob_u': [0.6, 0.4]
    })
    # P(Z=0|U=0) = 0.25, P(Z=1|U=0) = 0.35, P(Z=2|U=0) = 0.40
    # P(Z=0|U=1) = 0.40, P(Z=1|U=1) = 0.35, P(Z=2|U=1) = 0.25
    p_z_cond_u = pd.DataFrame({
        'U': [0, 0, 0, 0,
          1, 1, 1, 1],
        'Z': [0, 1, 2, 3,
          0, 1, 2, 3],
        'prob_z_cond_u': [0.1, 0.2, 0.3, 0.4,
                      0.4, 0.3, 0.2, 0.1]
    })
    # P(W=0|Z=0) = 1
    # P(W=1|Z=1) = 1
    # P(W=1|Z=2) = 1
    p_w_cond_z = pd.DataFrame({
        'Z': [0, 0,
          1, 1,
          2, 2,
          3, 3],
        'W': [0, 1,
          0, 1,
          0, 1,
          0, 1],
        'prob_w_cond_z': [1, 0,
                      1, 0, 
                      0, 1,
                      0, 1]
    })
    p_y_cond_uw = pd.DataFrame({
        'U': [0, 0,
          0, 0, 
          1, 1,
          1, 1],
        'W': [0, 0, 
          1, 1,
          0, 0,
          1, 1],
        'Y': [0, 1,
          0, 1,
          0, 1,
          0, 1],
        'prob_y_cond_uw': [0.20, 0.80,
                       0.40, 0.60,
                       0.60, 0.40, 
                       0.75, 0.25]
    })
    p_zwy = (p_u
         .merge(p_z_cond_u)
         .merge(p_w_cond_z)
         .merge(p_y_cond_uw)
         .assign(prob = lambda i: i.prob_u * i.prob_z_cond_u * i.prob_w_cond_z * i.prob_y_cond_uw)
         .drop(['prob_u', 'prob_z_cond_u', 'prob_w_cond_z', 'prob_y_cond_uw'], axis = 1)
         .groupby(['Z', 'W', 'Y'])
         .sum()['prob']
         .reset_index()
        )
    p_y_cond_w = (p_u
              .merge(p_y_cond_uw)
              .assign(prob = lambda i: i.prob_u * i.prob_y_cond_uw)
              .groupby(['W', 'Y'])
              .sum()['prob']
	      .reset_index()
              )
    dag = DAG()
    dag.from_structure('U -> Z, U -> Y, Z -> W, W -> Y')
    problem = causalProblem(dag, {'Z': 4})
    problem.set_ate('W', 'Y')
    # Explictly setting these strata to 0
    problem.set_p_to_zero(problem.query('W(Z=0)=1'))
    problem.set_p_to_zero(problem.query('W(Z=1)=1'))
    problem.set_p_to_zero(problem.query('W(Z=2)=0'))
    problem.set_p_to_zero(problem.query('W(Z=3)=0'))
    #problem.set_p_to_zero([ i[1][0] for i in problem.query('W(Z=0)=1')]) 
    #problem.set_p_to_zero([ i[1][0] for i in problem.query('W(Z=1)=1')])
    #problem.set_p_to_zero([ i[1][0] for i in problem.query('W(Z=2)=0')])
    #problem.set_p_to_zero([ i[1][0] for i in problem.query('W(Z=3)=0')])
    #problem.load_data(p_z_cond_u.rename({'prob_z_cond_u': 'prob'}, axis = 1), cond = ['U'])
    problem.load_data(p_zwy)
    # Explictly setting the remaining stratum to 0
    # This step is not necessary, but it might make sense to get a faster solution
#    problem.add_constraint(problem.query('W(Z=0)=0&W(Z=1)=0&W(Z=2)=1&W(Z=3)=1') - Query(1))
#    for u in range(2):
#        for z in range(4):
#            problem.add_constraint(problem.query(f'Z(U={u})={z}') - Query(p_z_cond_u.loc[lambda i: (i.U == u) & (i.Z == z)].iloc[0]['prob_z_cond_u']))
#    problem.add_constraint(problem.query('W(Z=0)=1'))  # prob of this expression == 0
#    problem.add_constraint(problem.query('W(Z=1)=1'))  # prob of this expression == 0
#    problem.add_constraint(problem.query('W(Z=2)=0'))  # prob of this expression == 0
#    problem.add_constraint(problem.query('W(Z=3)=0'))  # prob of this expression == 0
    problem.add_prob_constraints()
    program = problem.write_program()
    program.to_pip('rdd_no_simp.pip')
    program.simplify_linear()
    program.to_pip('rdd_simp.pip')


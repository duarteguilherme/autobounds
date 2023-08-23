from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io
from copy import deepcopy

#### Creating DAGs

# Standard IV DAG
dag = DAG()
dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")

# Z causes X and Y -- DAG
dag2 = DAG()
dag2.from_structure("Z -> Y, Z -> X, X -> Y, U -> X, U -> Y", unob = "U")


# We will create three scenarios -- 
# a) Overly cautious -- where one does not assume away Z does not cause Y
overly_cautious_problem = causalProblem(dag2)

# b) Standard IV problem
just_problem = causalProblem(dag)

# c) Overconfident -- problem is similar to B, but Z is assumed to affect monotonically X
overconfident_problem = causalProblem(dag)
overconfident_problem.set_p_to_zero( # Setting monotonicity
        [x[1][0] for x in overconfident_problem.query('X(Z=0)=1&X(Z=1)=0')]
        )


# Adding observational data and axioms of probability constraints for each case
just_problem.load_data('data/iv.csv')
just_problem.add_prob_constraints()
overly_cautious_problem.load_data('data/iv.csv')
overly_cautious_problem.add_prob_constraints()
overconfident_problem.load_data('data/iv.csv')
overconfident_problem.add_prob_constraints()


# Creating similar objects for LATE. Standard objects will be set to ATE.
just_problem_late = deepcopy(just_problem) 
overly_cautious_late = deepcopy(overly_cautious_problem) 
overconfident_late = deepcopy(overconfident_problem) 

# Adding estimands - ATE
just_problem.set_ate('X','Y')
overly_cautious_problem.set_ate('X','Y')
overconfident_problem.set_ate('X','Y')



# Adding estimates - LATE
just_problem_late.set_estimand(
        just_problem_late.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') -
        just_problem_late.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0'),
        div = just_problem.query('X(Z=1)=1&X(Z=0)=0'))
overly_cautious_late.set_estimand(
            overly_cautious_late.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') -
            overly_cautious_late.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0'),
            div = overly_cautious_late.query('X(Z=1)=1&X(Z=0)=0'))
overconfident_late.set_estimand(
            overconfident_late.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') -
            overconfident_late.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0'),
            div = overconfident_late.query('X(Z=1)=1&X(Z=0)=0'))


# Writing optimization programs
just_prog_ate = just_problem.write_program()
overly_prog_ate = overly_cautious_problem.write_program()
overconfident_prog_ate = overconfident_problem.write_program()

just_prog_late = just_problem_late.write_program()
overly_prog_late = overly_cautious_late.write_program()
overconfident_prog_late = overconfident_late.write_program()


just_prog_ate.run_couenne(filename = 'results/iv_ate_just.csv')
overly_prog_ate.run_couenne(filename = 'results/iv_ate_cautious.csv')
overconfident_prog_ate.run_couenne()

just_prog_late.run_couenne(filename = 'results/iv_late_just.csv')
overly_prog_late.run_couenne(filename = 'results/iv_late_cautious.csv')
overconfident_prog_late.run_couenne()


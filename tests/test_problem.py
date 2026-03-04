from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.Parser import *
from autobounds.autobounds.causalProblem import *
# 
klm_model = DAG("D -> Y, D -> M, U -> M, U -> Y, M -> Y", unob="U")

#def test_main():
#    prob_klm_relax_monotonicty_0_05 = causalProblem(klm_model)
#    prob_klm_relax_monotonicty_0_05.safe_min = 0.02
#
#    # add assumptions
#    with respect_to(prob_klm_relax_monotonicty_0_05):
#        # read data
#        read_data(data, cond=["M"], inference=True)
#        # estimand
#        set_ate("D", "Y", cond="M=1")
#        
#        # mandatory reporting
#        force_used_without_stop = p('Y(M=0)=1')  
#        add_assumption(force_used_without_stop, '==', 0.00)  
#
#        # no anti-white bias (mediator monotonicity)
#        anti_white_stop = p("M(D=0)=1 & M(D=1)=0")  
#        add_assumption(anti_white_stop, '<=', 0.01)  
#
#        # racial nonseverity assumption
#        racial_stop = "M(D=0)=0 & M(D=1)=1"
#        always_stop = "M(D=0)=1 & M(D=1)=1"
#        for d in [0,1]:
#            for m in [0,1]:
#                average_potential_force_in_racial_stops = E(f"Y(D={d}, M={m})", cond=racial_stop) 
#                average_potential_force_in_always_stops = E(f"Y(D={d}, M={m})", cond=always_stop)
#                add_assumption(
#                    average_potential_force_in_racial_stops,
#                    "<=",
#                    average_potential_force_in_always_stops
#                )
#
#        # previous research from Gelman, Fagan, Kiss (2007) 
#        # quantity is equal to the probability of necessity (Pearl, 2009)
#        black_necessity = p("M(D=0)=0", cond = "D=1 & M(D=1)=1")
#        add_assumption(
#           black_necessity, "==", Q(0.32)
#        )
#
#        res_klm_relax_monotonicty_0_05 = solve(ci=True, maxtime=300, theta=0.01, limits=[-1,1], nsamples = 200, verbose_optimizer = True)

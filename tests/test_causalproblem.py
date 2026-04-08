import numpy as np
import pytest
from autobounds.autobounds.DAG import DAG
from autobounds.autobounds.causalProblem import *
from autobounds.autobounds.Q import Q, clean_list, compare_lists
import pandas as pd
import io
from copy import deepcopy




def test_conditional_data():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.set_estimand(x.p('Y(X=1)=1') - x.p('Y(X=0)=1'))
    x.load_data(datafile, cond = ['X'])
    z = x.write_program()
    assert 'objvar' in z.parameters
    assert 'X00.Y00' in z.parameters
    assert 'Z0' in z.parameters
    assert len(z.constraints) == 13
    assert z.constraints[5] ==  [['0.95', 'X00.Y00', 'Z0'], ['0.95', 'X00.Y01', 'Z0'], ['0.95', 'X01.Y00', 'Z0'], ['0.95', 'X01.Y01', 'Z0'], ['-0.05', 'X00.Y00', 'Z1'], ['-0.05', 'X00.Y01', 'Z1'], ['-0.05', 'X00.Y10', 'Z0'], ['-0.05', 'X00.Y10', 'Z1'], ['-0.05',   'X00.Y11', 'Z0'], ['-0.05', 'X00.Y11', 'Z1'], ['-0.05', 'X01.Y10', 'Z0'], ['-0.05', 'X01.Y11', 'Z0'], ['-0.05', 'X10.Y00', 'Z1'], ['-0.05', 'X10.Y01', 'Z1'], ['-0.05', 'X10.Y10', 'Z1'], ['-0.05', 'X10.Y11', 'Z1'], ['==']]



def test_set_ate():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    x.set_estimand(x.p('Y(X=1)=1&X=0') - x.p('Y(X=0)=1&X=0'), div = x.p('X=0'))
    x.set_ate('X','Y', cond = 'X=0')
#    print(x.constraints)
    assert compare_lists(x.constraints[-1], x.constraints[2]) # Comparing two ways of setting the ATE
    # Notice that laws of probability are added twice due to setting two estimands. 
    # This problem cannot se solved of course, because it has two objective functions
    # Must check if denominators are not being added twice either.


def test_add_constraints():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y, K -> X", unob = "U")
    x = causalProblem(y, {'X': 2})
    assert (1, 'Z0') in x.parameters
    x.set_p_to_zero(['Z0'])
    assert 'Z0' in x.zero_parameters
    x.add_constraint(Q([(-0.15, ['1']), (-0.15, ['1']), (1, ['X1111']), (-1, ['X1111', 'Z1']), (2, ['X1111'])]))
    assert [(-0.3, ['1']), (3, ['X1111']), (-1, ['X1111', 'Z1']), (1, ['=='])] in x.constraints
    x.add_constraint(Q([(1, ['X1110']), (-1, ['X1110', 'Z1']), (-1, ['X1110'])]))
    assert [(-1, ['X1110', 'Z1']), (1, ['=='])] in x.constraints


def test_respect_to():
    d = DAG('D -> Y')
    pro = causalProblem(d)
    with respect_to(pro):
        add_assumption(p('Y(D=1)=0&Y(D=0)=1'), '==', 0.0)
        set_estimand(E('Y(D=1)') - E('Y(D=0)'))
        load_data(raw =  pd.DataFrame({
                'D': [0,0,1,0,1,0,1,1,1,0,1,0,0,1],
                'Y': [0,1,1,0,0,0,1,0,1,1,0,0,1,1]
            })           )
#        print(solve())

def test_respect_to_restores_existing_globals():
    d = DAG('D -> Y')
    pro = causalProblem(d)

    original_p = "preexisting_symbol"
    globals()["p"] = original_p
    try:
        with respect_to(pro):
            assert callable(p)
        assert globals()["p"] == original_p
    finally:
        if globals().get("p", None) == original_p:
            del globals()["p"]


def test_solve():
    d = DAG('D -> Y')
    pro = causalProblem(d)
    df =  pd.DataFrame({
                'D': [0,0,1,0,1,0,1,1,1,0,1,0,0,1],
                'Y': [0,1,1,0,0,0,1,0,1,1,0,0,1,1]
            })           
    pro.load_data(raw = df)
    pro.set_ate('D','Y')
    solution = pro.solve()
    assert solution['point lb primal'] == pytest.approx(0.142857, rel=1e-3, abs=1e-4)


def test_solve_return_dgps():
    d = DAG('D -> Y')
    pro = causalProblem(d)
    df = pd.DataFrame({
        'D': [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        'Y': [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    })
    pro.load_data(raw=df)
    pro.set_ate('D', 'Y')
    solution = pro.solve(return_dgps=True, verbose_result=False, verbose_optimizer=False)
    assert "dgps" in solution
    assert solution["dgps"]["lower"]["status"] == "ok"
    assert solution["dgps"]["upper"]["status"] == "ok"
    assert "objvar" in solution["dgps"]["lower"]["values"]
    assert "objvar" in solution["dgps"]["upper"]["values"]


def test_solve_return_dgps_read_data():
    dag = DAG()
    dag.from_structure("Z -> D, D -> Y, U -> D, U -> Y", unob="U")
    pro = causalProblem(dag)
    df = pd.DataFrame({
        "Z": [0, 0, 0, 0, 1, 1, 1, 1],
        "D": [0, 0, 1, 1, 0, 1, 0, 1],
        "Y": [0, 1, 0, 1, 0, 1, 1, 1],
    })
    pro.read_data(raw=df)
    pro.set_ate("D", "Y")
    solution = pro.solve(return_dgps=True, verbose_result=False, verbose_optimizer=False)
    assert "dgps" in solution
    assert solution["dgps"]["lower"]["status"] == "ok"
    assert solution["dgps"]["upper"]["status"] == "ok"
    assert "objvar" in solution["dgps"]["lower"]["values"]
    assert "objvar" in solution["dgps"]["upper"]["values"]

def test_load_raw():
    d = DAG('D -> Y')
    pro = causalProblem(d)
    df =  pd.DataFrame({
                'D': np.random.randint(0, 2, size=10),
                'Y': np.random.randint(0, 2, size=10)
            })           
    pro.load_data(raw = df)

def test_e():
    d = DAG()
    d.from_structure('D -> Y')
    problem = causalProblem(d, {'Y': 4})
    assert (problem.E('Y(D=1)') == problem.p('Y(D=1)=1') + problem.p('Y(D=1)=2') * Q(2) + problem.p('Y(D=1)=3') * Q(3) )
    assert (problem.E('Y') == problem.p('Y=1') + problem.p('Y=2') * 2 + problem.p('Y=3') * 3 )



def test_add_assumption():
    d = DAG('D -> Y')
    problem = causalProblem(d)
    zero_query = problem.p('Y(D=1)=0&Y(D=0)=1')
    zero_terms = {term[1][0] for term in zero_query._event}
    problem.add_assumption(zero_query, '==', 0.0)
    assert zero_terms.issubset(problem.zero_parameters)


def test_continuous_outcome_load_data_discretizes_to_five_bins():
    problem = causalProblem(DAG("D -> Y"), continuous_outcome="Y")
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 10, "Y": np.linspace(10.0, 29.0, 20)})

    problem.load_data(raw=df)

    assert problem.get_bounder("default").number_values["Y"] == 5
    assert sorted(problem.get_bounder("default").datam["Y"].unique().tolist()) == [0, 1, 2, 3, 4]
    specs = problem._continuous_bin_specs
    assert len(specs) == 5
    assert [spec["bin"] for spec in specs] == [0, 1, 2, 3, 4]
    assert [spec["ymin"] for spec in specs] == pytest.approx([10.0, 14.0, 18.0, 22.0, 26.0])
    assert [spec["ymax"] for spec in specs] == pytest.approx([13.0, 17.0, 21.0, 25.0, 29.0])
    assert specs[0]["normalized_ymin"] == pytest.approx(0.0)
    assert specs[-1]["normalized_ymax"] == pytest.approx(1.0)


def test_continuous_outcome_solve_aggregates_thresholds_with_endpoint_gaps(monkeypatch):
    class _FakeThresholdProblem:
        def __init__(self, cutoff):
            self.cutoff = cutoff

        def solve(self, **kwargs):
            base = 0.1 * self.cutoff
            return {
                "point lb dual": base,
                "point ub dual": base + 0.05,
                "point lb primal": base,
                "point ub primal": base + 0.05,
            }

    monkeypatch.setattr(
        causalProblem,
        "_build_threshold_problem",
        lambda self, ind, dep, cutoff, direction: _FakeThresholdProblem(cutoff),
    )

    problem = causalProblem(DAG("D -> Y"), continuous_outcome=True, continuous_method="ymax_ymin")
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 10, "Y": np.linspace(10.0, 29.0, 20)})
    problem.load_data(raw=df)
    out = problem.solve(verbose_result=False)

    expected_lower_weights = [4.0, 4.0, 4.0, 4.0]
    expected_upper_weights = [4.0, 4.0, 4.0, 4.0]
    expected_lb = sum(w * 0.1 * cutoff for cutoff, w in zip([1, 2, 3, 4], expected_lower_weights))
    expected_ub = sum(w * (0.1 * cutoff + 0.05) for cutoff, w in zip([1, 2, 3, 4], expected_upper_weights))
    assert out["point lb dual"] == pytest.approx(expected_lb)
    assert out["point ub dual"] == pytest.approx(expected_ub)
    assert out["point lb primal"] == pytest.approx(expected_lb)
    assert out["point ub primal"] == pytest.approx(expected_ub)
    assert out["continuous_outcome"] is True
    assert out["continuous_lower_representatives"] == pytest.approx([10.0, 14.0, 18.0, 22.0, 26.0])
    assert out["continuous_upper_representatives"] == pytest.approx([13.0, 17.0, 21.0, 25.0, 29.0])
    assert sorted(out["threshold_results"].keys()) == [1, 2, 3, 4]
    assert out["threshold_results"][1]["lower_weight"] == pytest.approx(4.0)
    assert out["threshold_results"][1]["upper_weight"] == pytest.approx(4.0)


def test_continuous_outcome_solve_aggregates_thresholds_with_midpoints(monkeypatch):
    class _FakeThresholdProblem:
        def __init__(self, cutoff):
            self.cutoff = cutoff

        def solve(self, **kwargs):
            base = 0.1 * self.cutoff
            return {
                "point lb dual": base,
                "point ub dual": base + 0.05,
                "point lb primal": base,
                "point ub primal": base + 0.05,
            }

    monkeypatch.setattr(
        causalProblem,
        "_build_threshold_problem",
        lambda self, ind, dep, cutoff, direction: _FakeThresholdProblem(cutoff),
    )

    problem = causalProblem(DAG("D -> Y"), continuous_outcome=True, continuous_method="midpoint")
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 10, "Y": np.linspace(10.0, 29.0, 20)})
    problem.load_data(raw=df)
    out = problem.solve(verbose_result=False)

    expected_reps = [11.5, 15.5, 19.5, 23.5, 27.5]
    expected_weights = [4.0, 4.0, 4.0, 4.0]
    expected_lb = sum(w * 0.1 * cutoff for cutoff, w in zip([1, 2, 3, 4], expected_weights))
    expected_ub = sum(w * (0.1 * cutoff + 0.05) for cutoff, w in zip([1, 2, 3, 4], expected_weights))
    assert out["point lb dual"] == pytest.approx(expected_lb)
    assert out["point ub dual"] == pytest.approx(expected_ub)
    assert out["continuous_method"] == "midpoint"
    assert out["continuous_lower_representatives"] == pytest.approx(expected_reps)
    assert out["continuous_upper_representatives"] == pytest.approx(expected_reps)
    assert out["threshold_results"][1]["lower_weight"] == pytest.approx(4.0)
    assert out["threshold_results"][1]["upper_weight"] == pytest.approx(4.0)


def test_continuous_outcome_conservative_ate_combines_components(monkeypatch):
    class _FakeThresholdProblem:
        def __init__(self, cutoff, treatment_value):
            self.cutoff = cutoff
            self.treatment_value = treatment_value

        def solve(self, **kwargs):
            base = 0.2 * self.cutoff + 0.5 * self.treatment_value
            return {
                "point lb dual": base,
                "point ub dual": base + 0.1,
                "point lb primal": base,
                "point ub primal": base + 0.1,
            }

    monkeypatch.setattr(
        causalProblem,
        "_build_threshold_component_problem",
        lambda self, ind, dep, cutoff, treatment_value, cond=None: _FakeThresholdProblem(
            cutoff, treatment_value
        ),
    )

    problem = causalProblem(DAG("D -> Y"), continuous_outcome=True, continuous_method="conservative")
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 10, "Y": np.linspace(10.0, 29.0, 20)})
    problem.load_data(raw=df)
    out = problem.solve(verbose_result=False)

    weights = [4.0, 4.0, 4.0, 4.0]
    ey1_lb = sum(w * (0.2 * cutoff + 0.5) for cutoff, w in zip([1, 2, 3, 4], weights))
    ey1_ub = sum(w * (0.2 * cutoff + 0.6) for cutoff, w in zip([1, 2, 3, 4], weights))
    ey0_lb = sum(w * (0.2 * cutoff + 0.0) for cutoff, w in zip([1, 2, 3, 4], weights))
    ey0_ub = sum(w * (0.2 * cutoff + 0.1) for cutoff, w in zip([1, 2, 3, 4], weights))

    assert out["point lb dual"] == pytest.approx(ey1_lb - ey0_ub)
    assert out["point ub dual"] == pytest.approx(ey1_ub - ey0_lb)
    assert out["point lb primal"] == pytest.approx(ey1_lb - ey0_ub)
    assert out["point ub primal"] == pytest.approx(ey1_ub - ey0_lb)
    assert out["continuous_method"] == "conservative"
    assert set(out["component_results"].keys()) == {0, 1}


def test_continuous_outcome_conservative_ate_combines_ci_and_dgps(monkeypatch):
    class _FakeThresholdProblem:
        def __init__(self, cutoff, treatment_value):
            self.cutoff = cutoff
            self.treatment_value = treatment_value

        def solve(self, **kwargs):
            base = 0.2 * self.cutoff + 0.5 * self.treatment_value
            out = {
                "point lb dual": base,
                "point ub dual": base + 0.1,
                "point lb primal": base,
                "point ub primal": base + 0.1,
            }
            if kwargs.get("ci", False):
                out["2.5% lb bounds"] = base - 0.01
                out["1% lb bounds"] = base - 0.02
                out["97.5% ub bounds"] = base + 0.11
                out["99% ub bounds"] = base + 0.12
                out["ci method"] = "recentered_subsampling"
                out["ci workers"] = 3
            if kwargs.get("return_dgps", False):
                out["dgps"] = {
                    "lower": {"status": f"lower_{self.treatment_value}_{self.cutoff}"},
                    "upper": {"status": f"upper_{self.treatment_value}_{self.cutoff}"},
                }
            return out

    monkeypatch.setattr(
        causalProblem,
        "_build_threshold_component_problem",
        lambda self, ind, dep, cutoff, treatment_value, cond=None: _FakeThresholdProblem(
            cutoff, treatment_value
        ),
    )

    problem = causalProblem(DAG("D -> Y"), continuous_outcome=True, continuous_method="conservative")
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 10, "Y": np.linspace(10.0, 29.0, 20)})
    problem.load_data(raw=df)
    out = problem.solve(ci=True, return_dgps=True, verbose_result=False)

    weights = [4.0, 4.0, 4.0, 4.0]
    ey1_lb_ci = sum(w * (0.2 * cutoff + 0.5 - 0.01) for cutoff, w in zip([1, 2, 3, 4], weights))
    ey1_ub_ci = sum(w * (0.2 * cutoff + 0.5 + 0.11) for cutoff, w in zip([1, 2, 3, 4], weights))
    ey0_lb_ci = sum(w * (0.2 * cutoff - 0.01) for cutoff, w in zip([1, 2, 3, 4], weights))
    ey0_ub_ci = sum(w * (0.2 * cutoff + 0.11) for cutoff, w in zip([1, 2, 3, 4], weights))

    assert out["2.5% lb bounds"] == pytest.approx(ey1_lb_ci - ey0_ub_ci)
    assert out["97.5% ub bounds"] == pytest.approx(ey1_ub_ci - ey0_lb_ci)
    assert out["ci method"] == "recentered_subsampling"
    assert out["ci workers"] == 3
    assert out["dgps"]["lower"]["status"] == "continuous_conservative_aggregate"
    assert out["dgps"]["upper"]["status"] == "continuous_conservative_aggregate"


def test_continuous_outcome_solve_aggregates_ci_and_dgps(monkeypatch):
    class _FakeThresholdProblem:
        def __init__(self, cutoff):
            self.cutoff = cutoff

        def solve(self, **kwargs):
            base = 0.1 * self.cutoff
            out = {
                "point lb dual": base,
                "point ub dual": base + 0.05,
                "point lb primal": base,
                "point ub primal": base + 0.05,
            }
            if kwargs.get("ci", False):
                out["2.5% lb bounds"] = base - 0.01
                out["1% lb bounds"] = base - 0.02
                out["97.5% ub bounds"] = base + 0.06
                out["99% ub bounds"] = base + 0.07
                out["ci method"] = "recentered_subsampling"
                out["ci workers"] = 2
            if kwargs.get("return_dgps", False):
                out["dgps"] = {
                    "lower": {"status": f"lower_{self.cutoff}"},
                    "upper": {"status": f"upper_{self.cutoff}"},
                }
            return out

    monkeypatch.setattr(
        causalProblem,
        "_build_threshold_problem",
        lambda self, ind, dep, cutoff, direction: _FakeThresholdProblem(cutoff),
    )

    problem = causalProblem(DAG("D -> Y"), continuous_outcome=True)
    problem.set_ate("D", "Y")
    df = pd.DataFrame({"D": [0, 1] * 10, "Y": np.linspace(10.0, 29.0, 20)})
    problem.load_data(raw=df)
    out = problem.solve(ci=True, return_dgps=True, verbose_result=False)

    expected_lower_weights = [4.0, 4.0, 4.0, 4.0]
    expected_upper_weights = [4.0, 4.0, 4.0, 4.0]
    expected_lb_ci = sum(w * (0.1 * cutoff - 0.01) for cutoff, w in zip([1, 2, 3, 4], expected_lower_weights))
    expected_ub_ci = sum(w * (0.1 * cutoff + 0.06) for cutoff, w in zip([1, 2, 3, 4], expected_upper_weights))
    assert out["2.5% lb bounds"] == pytest.approx(expected_lb_ci)
    assert out["97.5% ub bounds"] == pytest.approx(expected_ub_ci)
    assert out["ci method"] == "recentered_subsampling"
    assert out["ci workers"] == 2
    assert out["dgps"]["lower"]["status"] == "continuous_threshold_aggregate"
    assert out["dgps"]["upper"]["status"] == "continuous_threshold_aggregate"
    assert len(out["dgps"]["lower"]["thresholds"]) == 4
    assert out["dgps"]["lower"]["thresholds"][0]["dgps"]["status"] == "lower_1"
    assert out["dgps"]["upper"]["thresholds"][-1]["dgps"]["status"] == "upper_4"
    assert out["dgps"]["lower"]["thresholds"][0]["weight"] == pytest.approx(4.0)
    assert out["dgps"]["upper"]["thresholds"][0]["weight"] == pytest.approx(4.0)






def test_solve_kl():
    ns = 1000
    K = 8
    o = 0.125
    alpha = 0.05
    result = solve_kl_p(ns=ns, K = K, o = o, alpha = alpha)
    assert result[0] > 0.09
    assert result[1] < 0.17

def test_solve_kl_bug_log():
    ns = 20
    K = 8
    o = 0.0001
    alpha = 0.05
    result = solve_kl_p(ns=ns, K = K, o = o, alpha = alpha)
    assert result[0] == 0



def test_load_data_do(tmp_path):
    df_y_do_x = pd.DataFrame({
        'X': [0,0,1,1],
        'Y': [0,1,0,1],
        'prob': [0.4, 0.6, 0.1, 0.9]
        })
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = 'U')
    problem = causalProblem(dag)
    problem.load_data(df_y_do_x, do = ['X'])
    problem.set_ate('X','Y')
    program = problem.write_program()
    program.to_pip(str(tmp_path / 'test_do.pip'))
    res = program.run_scip()
    assert res[0]['primal'] == 0.3


def test_solve_gaussian():
    res = solve_gaussian(nr = 100, o = [0.25, 0.25, 0.25, 0.25] , alpha = 0.05)

def test_load_data_gaussian():
    datafile = io.StringIO('''X,Y,prob
    0,0,0.25
    0,1,0.25
    1,0,0.25
    1,1,0.25''')
    y = DAG()
    y.from_structure("U -> X, U -> Y", unob = 'U')
    x = causalProblem(y, {'X': 2})
    x.load_data_gaussian(datafile, N = 1000)
    p00_problem, p01_problem, p10_problem, p11_problem = [deepcopy(x) for i in range(4) ]
    p00_problem.set_estimand(x.p('X=0&Y=0'))
#    p01_problem.set_estimand(x.p('X=0&Y=1'))
#    p10_problem.set_estimand(x.p('X=1&Y=0'))
#    p11_problem.set_estimand(x.p('X=1&Y=1'))
#    p00 = p00_problem.write_program().run_couenne()
#    p01 = p01_problem.write_program().run_couenne()
#    p10 = p10_problem.write_program().run_couenne()
#    p11 = p11_problem.write_program().run_couenne()
#    p00_problem.write_program().to_pip('/home/beta/gauss.pip')


def test_load_pandas():
    data = pd.DataFrame({
        'X': [0,0,0,0,1,1,1,1],
        'Y': [0,0,1,1,0,0,1,1],
        'Z': [0,1,0,1,0,1,0,1],
        'prob': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        })
    y = DAG()
    y.from_structure("Z -> Y, X -> Y, U -> X, U -> Y", unob = 'U')
    x = causalProblem(y, {'X': 2})
    x.load_data(data) # No need to assert

def test_load_data_kl():
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
    x.load_data_kl(datafile, N = 1000)
    assert -1 * x.constraints[-1][-2][0] == solve_kl_p(ns = 1000, K = 8, o = 0.125, alpha = 0.05)[1]




def test_check_constraints():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.load_data(datafile)
    x.check_constraints()
    assert (0.5, ['X00.Y00', '1']) in x.constraints[0]


def test_conditional_estimand():
    y = DAG()
    y.from_structure("X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    x.set_estimand(x.p('Y(X=1)=1') - x.p('Y(X=0)=1'), div = x.p('X=0'))
    assert Q(x.constraints[-1]) ==  Q('X0.Y01') + Q('X1.Y01') - Q('X0.Y10') - Q('X1.Y10') - ( x.p('X=0') * Q('objvar') ) + Q('==')


def test_causalproblem():
    y = DAG()
    y.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    x = causalProblem(y, {'X': 2})
    z = Parser(y)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    x.set_estimand(x.p('Y(X=1)=1') - x.p('Y(X=0)=1'))
    x.load_data(datafile)
    z = x.write_program()
    assert 'objvar' in z.parameters
    assert 'X00.Y00' in z.parameters
    assert len(z.constraints) == 10
#[['0.05'], ['0.5', 'X00.Y00'], ['0.5', 'X00.Y01'], ['0.5', 'X01.Y00'], ['0.5', 'X01.Y01'], ['==']]
    assert z.constraints[1] == [['X00.Y01'], ['X01.Y01'], ['X10.Y01'], ['X11.Y01'], ['-1', 'X00.Y10'], ['-1', 'X01.Y10'], ['-1', 'X10.Y10'], ['-1', 'X11.Y10'], ['-1', 'objvar'], ['==']]
    assert z.constraints[2] == [['0.5', 'X00.Y00'], ['0.5', 'X00.Y01'], ['0.5', 'X01.Y00'], ['0.5', 'X01.Y01'], ['-0.05'], ['==']]

def test_replace_first_nodes():
    assert replace_first_nodes([('Z0', 0.5), ('Z1', 0.5)], 
            (1, ['X00.Y10', 'Z0'])) == (0.5, ['X00.Y10', '1'])


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
    y = DAG()
    y.from_structure("Z -> Y, U -> Z, X -> Y, U -> Y, U -> X", unob = "U")
    x = causalProblem(y, {'X': 2})
    x.load_data(datafile2)

def test_transform_constraint():
    assert transform_constraint([(1, ['X00.Y01']), (1, ['X01.Y01']), (1, ['X10.Y01']),
        (1, ['X11.Y01']), (-1, ['X00.Y10']), (-1, ['X01.Y10']), 
        (-1, ['X10.Y10']), (-1, ['X11.Y10']), (-1, ['1', 'objvar'])]) == [['X00.Y01'], 
                ['X01.Y01'], ['X10.Y01'], ['X11.Y01'], ['-1', 'X00.Y10'],
                ['-1', 'X01.Y10'], ['-1', 'X10.Y10'], ['-1', 'X11.Y10'], ['-1', 'objvar']]
    model = DAG()
    model.from_structure("D -> Y, D -> M, M -> Y, U -> Y, U -> M", unob = "U")
    problem = causalProblem(model)
#    problem.set_p_to_zero([ x[1][0] for x in problem.p('M(D=0)=1&M(D=1)=0') ])
    problem.set_p_to_zero(problem.p('M(D=0)=1&M(D=1)=0'))
    problem.set_estimand(problem.p('Y(D=1)=1') - problem.p('Y(D=0)=1'),div = problem.p('M=1'))
    problem.constraints[-1]
    program = problem.write_program()
    assert program.constraints[-1] == [['M00.Y0010'], ['M00.Y0011'], ['M00.Y0110'], ['M00.Y0111'], ['M01.Y0001'], ['M01.Y0011'], ['M01.Y0101'], ['M01.Y0111'], ['M11.Y0001'], ['M11.Y0011'], ['M11.Y1001'], ['M11.Y1011'], ['-1', 'M00.Y1000'], ['-1', 'M00.Y1001'], ['-1', 'M00.Y1100'], ['-1', 'M00.Y1101'], ['-1', 'M01.Y1000'], ['-1', 'M01.Y1010'], ['-1', 'M01.Y1100'], ['-1', 'M01.Y1110'], ['-1', 'M11.Y0100'], ['-1', 'M11.Y0110'], ['-1', 'M11.Y1100'], ['-1', 'M11.Y1110'], ['-1', 'D0', 'M11.Y0000', 'objvar'], ['-1', 'D0', 'M11.Y0001', 'objvar'], ['-1', 'D0','M11.Y0010', 'objvar'], ['-1', 'D0', 'M11.Y0011', 'objvar'], ['-1', 'D0', 'M11.Y0100', 'objvar'], ['-1', 'D0', 'M11.Y0101', 'objvar'], ['-1', 'D0', 'M11.Y0110', 'objvar'], ['-1', 'D0', 'M11.Y0111', 'objvar'], ['-1', 'D0', 'M11.Y1000', 'objvar'], ['-1', 'D0', 'M11.Y1001', 'objvar'], ['-1', 'D0', 'M11.Y1010', 'objvar'], ['-1', 'D0', 'M11.Y1011', 'objvar'], ['-1', 'D0', 'M11.Y1100', 'objvar'], ['-1', 'D0', 'M11.Y1101', 'objvar'], ['-1', 'D0', 'M11.Y1110', 'objvar'], ['-1', 'D0', 'M11.Y1111', 'objvar'], ['-1', 'D1', 'M01.Y0000', 'objvar'], ['-1', 'D1', 'M01.Y0001', 'objvar'], ['-1', 'D1', 'M01.Y0010', 'objvar'], ['-1', 'D1', 'M01.Y0011', 'objvar'], ['-1', 'D1', 'M01.Y0100', 'objvar'], ['-1', 'D1', 'M01.Y0101', 'objvar'], ['-1', 'D1', 'M01.Y0110', 'objvar'], ['-1', 'D1', 'M01.Y0111', 'objvar'], ['-1', 'D1', 'M01.Y1000', 'objvar'], ['-1', 'D1', 'M01.Y1001', 'objvar'], ['-1', 'D1', 'M01.Y1010', 'objvar'], ['-1', 'D1', 'M01.Y1011', 'objvar'], ['-1', 'D1', 'M01.Y1100', 'objvar'], ['-1', 'D1', 'M01.Y1101', 'objvar'], ['-1', 'D1', 'M01.Y1110', 'objvar'], ['-1', 'D1', 'M01.Y1111', 'objvar'], ['-1', 'D1', 'M11.Y0000', 'objvar'], ['-1', 'D1', 'M11.Y0001', 'objvar'], ['-1', 'D1', 'M11.Y0010', 'objvar'], ['-1', 'D1', 'M11.Y0011', 'objvar'], ['-1', 'D1', 'M11.Y0100', 'objvar'], ['-1', 'D1', 'M11.Y0101', 'objvar'], ['-1', 'D1', 'M11.Y0110', 'objvar'], ['-1', 'D1', 'M11.Y0111', 'objvar'], ['-1', 'D1', 'M11.Y1000', 'objvar'], ['-1', 'D1', 'M11.Y1001','objvar'], ['-1', 'D1', 'M11.Y1010', 'objvar'], ['-1', 'D1', 'M11.Y1011', 'objvar'], ['-1', 'D1', 'M11.Y1100', 'objvar'], ['-1', 'D1', 'M11.Y1101', 'objvar'], ['-1', 'D1', 'M11.Y1110', 'objvar'], ['-1', 'D1', 'M11.Y1111', 'objvar'], ['==']]

from autobounds.causalProblem import causalProblem
from autobounds.DAG import DAG
from autobounds.Program import change_constraint_parameter_value
from autobounds.Q import Q
import io
import time
import pandas as pd
import importlib
import multiprocessing
import pytest

ProgramModule = importlib.import_module("autobounds.Program")
ScipModule = importlib.import_module("autobounds.backends.scip")


def _worker_without_results(*args):
    """Simulate a worker that exits before producing any solver output."""
    return None


def test_failed_workers_are_joined_and_temporary_files_removed(monkeypatch, tmp_path):
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip('SCIP model worker integration requires fork')
    context = multiprocessing.get_context('fork')
    processes = []

    class RecordingContext:
        def Process(self, **kwargs):
            process = context.Process(**kwargs)
            processes.append(process)
            return process

    monkeypatch.setattr(ScipModule, 'get_context', lambda *args: RecordingContext())
    monkeypatch.setattr(ScipModule, 'solve_scip', _worker_without_results)
    monkeypatch.setattr(ScipModule.tempfile, 'tempdir', str(tmp_path))
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], ['==']]]
    with pytest.raises(RuntimeError, match='without a recognized final status'):
        program.run_scip(verbose=False, maxtime=1, return_dgps=True)
    assert len(processes) == 2
    assert all(not process.is_alive() and process.exitcode is not None
               for process in processes)
    assert list(tmp_path.iterdir()) == []


#

def test_optimizers():
    const = [['Y01'], ['-1', 'Y10'], ['X10', 'X11', 'Y11'], ['-1', 'objvar'], ['==']]
    assert ['-0.2'] in change_constraint_parameter_value(const, 'Y10', 0.20)
    assert ['0.31', 'X10', 'Y11'] in change_constraint_parameter_value(const, 'X11', 0.31)
    assert const[1] == ['-1', 'Y10']
    assert const[2] == ['X10', 'X11', 'Y11']

def test_program_scip_infeasibility():
    dag = DAG()
    dag.from_structure("D -> Y")
    problem = causalProblem(dag)
    problem.set_estimand(problem.p('Y(D=1)=1') - problem.p('Y(D=0)=1'))
    problem.add_prob_constraints()
    problem.add_constraint(problem.p('Y(D=0)=0&Y(D=1)=1') - Q(2))
    z = problem.write_program()
    z.optimize_remove_numeric_lines()
    res = z.run_scip(maxtime = 5)
    assert res[0]['end'] == 0
    assert res[1]['end'] == 0

def test_program_scip_infeasibility2():
    df = pd.DataFrame(
            {'Z': [0,0,0,0,1,1,1,1],
             'X': [0,0,1,1,0,0,1,1],
             'Y': [0,1,0,1,0,1,0,1],
          'prob': [0.066, 0.031, 0.377, 0.176, 0.063, 0.198, 0.021, 0.068 ]
             })
    dag = DAG()
    dag.from_structure('Z -> X, X -> Y, U -> X, U -> Y', unob = 'U')
    pro = causalProblem(dag)
    pro.load_data(df)
    pro.add_prob_constraints()
    pro.set_ate('X','Y', cond = 'X(Z=1)=1&X(Z=0)=0')
    pro.add_constraint(pro.p('X(Z=1)=1&X(Z=0)=0') - Q(0.0001), '>=')
    program = pro.write_program()
    program.run_scip()


### ISSUE: running with max time does not exit
def test_program_scip_time():
    dag = DAG()
    dag.from_structure("Z -> Y, X -> Y, U -> X, U -> Z", unob = "U")
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
    problem.set_estimand(problem.p('Y(X=1)=1') - problem.p('Y(X=0)=1'))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_scip(maxtime = 5)



#
def test_program_scip_rejects_unnormalized_probabilities():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,Z,prob
    0,0,0,1.0
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.p('Y(X=1)=1') - problem.p('Y(X=0)=1'))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_scip()
    assert res[0]['end'] == 0
    assert res[1]['end'] == 0


def test_program_pip_conf():
    dag = DAG()
    dag.from_structure("D -> Y, U -> D, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''D,Y,prob
    0,0,0.25
    0,1,0.25
    1,0,0.25
    1,1,0.25
    ''')
    problem.set_ate('D','Y')
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    res = z.run_scip()


# #
def test_program_parallel():
    dag = DAG()
    dag.from_structure("W -> X, W -> Y, W -> P, X -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''X,Y,P,prob
    0,0,0,0.05
    0,0,1,0.05
    0,1,0,0.1
    0,1,1,0.1
    1,0,0,0.15
    1,0,1,0.15
    1,1,0,0.2
    1,1,1,0.2''')
    problem.set_estimand(problem.p('Y(X=1)=1') - problem.p('Y(X=0)=1'))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
#    res = z.run_pyomo('ipopt', parallel = True, verbose = False)
#    assert res[0] < -0.08
#    assert res[0] < -0.08
#    assert res[1] > -0.1
#    assert res[1] > -0.1


def test_couenne_parse():
    dag = DAG()
    dag.from_structure("A -> Y, U -> A, U -> Y", unob = "U")
    problem = causalProblem(dag) 
    datafile = io.StringIO('''A,Y,prob
    0,0,0.13
    0,1,0.27
    1,0,0.2
    1,1,0.4''')
    problem.load_data(datafile)
    problem.set_ate('A','Y')
    program = problem.write_program()
    # lower, upper, theta, epsilon = program.run_couenne()
    # assert theta == 1
    # assert epsilon == 0
    # assert upper['primal'] < 0.54
    # assert upper['primal'] > 0.52
    # assert upper['dual'] < 0.54
    # assert upper['dual'] > 0.52
    # assert lower['primal'] < -0.46
    # assert lower['primal'] > -0.48
    # assert lower['dual'] < -0.46
    # assert lower['dual'] > -0.48


def test_couenne_threshold():
    dag = DAG()
    dag.from_structure("A -> B, B -> Y, U -> A, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile = io.StringIO('''A,B,Y,prob
    0,0,0,0.1
    0,0,1,0.1
    0,1,0,0.13
    0,1,1,0.1
    1,0,0,0.12
    1,0,1,0.15
    1,1,0,0.1
    1,1,1,0.2''')
    problem.load_data(datafile)
    problem.set_ate('A','Y')
    program = problem.write_program()
#    result = program.run_couenne(theta = 0.4, epsilon = 1)



def test_numeric_lines():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    datafile1 = io.StringIO('''X,A,prob
    0,1,0.230271252339654
    1,1,0.299527260157004''')
    datafile2 = io.StringIO('''Y,B,prob
    0,1,0.317930571936706
    1,1,0.176816814870372''')
    datafile3 = io.StringIO('''X,Y,A,B,prob
    0,0,1,1,0.087024102667622
    1,0,1,1,0.171021677876195
    0,1,1,1,0.0777025072996138
    1,1,1,1,0.0516439605484204''')
    datafile4 = io.StringIO('''A,B,prob
    0,0,0.362846349088114
    1,0,0.142406264104807
    0,1,0.107355138415228
    1,1,0.387392248391851''')
    dag = DAG()
    dag.from_structure("X -> Y, A -> B, Y -> B")
    problem = causalProblem(dag)
    problem.load_data(datafile1, optimize = True)
    problem.load_data(datafile2, optimize = True)
    problem.load_data(datafile3, optimize = True)
    problem.load_data(datafile4, optimize = True)
    problem.set_estimand(problem.p('Y(X=1)=1') - problem.p('Y(X=0)=1'))
    problem.add_prob_constraints()
    program = problem.write_program()


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
    problem.set_estimand(problem.p('Y(X=1)=1') + problem.p('Y(X=0)=1', sign = -1))
    problem.load_data(datafile)
    problem.add_prob_constraints()
    z = problem.write_program()
    # b = z.run_pyomo(verbose=False)
    # assert b[0] <= -0.48
    # assert b[0] >= -0.52
    # assert b[1] <= 0.52
    # assert b[1] >= 0.48


def test_program_iv_returns_dgps():
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
    problem.set_ate('X', 'Y')
    problem.load_data(datafile)
    problem.add_prob_constraints()
    program = problem.write_program()

    bounds, dgps = program.run_scip(return_dgps = True, verbose = False)

    assert isinstance(bounds, tuple)
    assert isinstance(dgps, dict)
    assert "lower" in dgps
    assert "upper" in dgps
    assert dgps["lower"].get("status") == "ok"
    assert dgps["upper"].get("status") == "ok"
    assert "values" in dgps["lower"]
    assert "values" in dgps["upper"]
    assert "objvar" in dgps["lower"]["values"]
    assert "objvar" in dgps["upper"]["values"]
    assert dgps["lower"]["values"]["objvar"] <= bounds[1]["primal"] + 1e-8
    assert dgps["upper"]["values"]["objvar"] >= bounds[0]["primal"] - 1e-8


def test_run_scip_return_dgps_handles_invalid_json(monkeypatch):
    class DummyProcess:
        def start(self):
            return None

        def is_alive(self):
            return False

        def terminate(self):
            return None

    class DummyContext:
        def Process(self, target=None, args=()):
            return DummyProcess()

    monkeypatch.setattr(ScipModule, "get_context", lambda *args, **kwargs: DummyContext())
    monkeypatch.setattr(
        ScipModule,
        "parse_bounds_scip",
        lambda *args, **kwargs: (
            {"primal": -0.5, "dual": -0.5, "time": 0.0, "end": 1},
            {"primal": 0.5, "dual": 0.5, "time": 0.0, "end": 1},
            1.0,
            0.0,
        ),
    )

    program = ProgramModule.Program()
    program.parameters = ["objvar"]
    program.constraints = [[["objvar"], ["=="]]]
    bounds, dgps = program.run_scip(return_dgps=True, verbose=False)

    assert bounds[0]["dual"] == -0.5
    assert bounds[1]["dual"] == 0.5
    assert dgps["lower"]["status"] == "parse_error"
    assert dgps["upper"]["status"] == "parse_error"


@pytest.mark.parametrize('return_dgps', [False, True])
def test_get_bounds_scip_returns_worker_results(return_dgps):
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], ['-0.3'], ['==']]]
    with pytest.raises(RuntimeError, match='Run run_scip'):
        program.get_bounds_scip()
    result = program.run_scip(verbose=False, return_dgps=return_dgps)
    bounds = result[0] if return_dgps else result
    for actual, expected in zip(program.get_bounds_scip(), bounds[:2]):
        assert actual['primal'] == pytest.approx(0.3)
        assert actual['dual'] == pytest.approx(0.3)
        assert actual == {key: expected[key] for key in ('dual', 'primal')}
    program.get_bounds_scip()[0]['dual'] = 999
    assert program.get_bounds_scip()[0]['dual'] == pytest.approx(0.3)


@pytest.mark.parametrize('terms,relation,feasible', [
    ([['0.1'], ['0.2'], ['-0.3']], '==', True),
    ([['0.01']], '==', False),
    ([['-0.01']], '>=', False),
    ([['0.01']], '<=', False),
    ([['-2', '3'], ['6']], '==', True),
    ([[]], '==', False),
    ([[], ['-1']], '==', True),
    ([], '==', True),
    ([['-2']], '<=', True),
    ([['2']], '>=', True),
])
def test_constant_constraints_preserve_feasibility(terms, relation, feasible):
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.constraints = [terms + [[relation]],
                           [['objvar'], ['-0.25'], ['==']]]
    program.optimize_remove_numeric_lines()
    result = program.run_scip(verbose=False)
    for endpoint in result[:2]:
        if feasible:
            assert endpoint['primal'] == pytest.approx(0.25)
            assert endpoint['dual'] == pytest.approx(0.25)
        else:
            assert endpoint['end'] == 0


def test_zero_parameter_substitution_preserves_infeasibility():
    from autobounds import Bounder
    problem = Bounder(DAG('X -> Y'))
    event = problem.p('Y(X=0)=1')
    problem.set_p_to_zero(event)
    problem.add_constraint(event - Q(0.25))
    problem.set_estimand(problem.p('Y(X=1)=1'))
    result = problem.write_program().run_scip(verbose=False)
    assert all(endpoint['end'] == 0 for endpoint in result[:2])


@pytest.mark.parametrize('method', ['run_couenne', 'run_pyomo', 'to_obj_pyomo', 'to_cip'])
def test_retired_backends_fail_explicitly(method):
    with pytest.raises(NotImplementedError, match='use'):
        getattr(ProgramModule.Program(), method)()


def test_spawn_only_platform_fails_before_creating_solver_files(monkeypatch, tmp_path):
    def unsupported_context(method):
        raise ValueError('fork unavailable')
    monkeypatch.setattr(ScipModule, 'get_context', unsupported_context)
    monkeypatch.setattr(ScipModule.tempfile, 'tempdir', str(tmp_path))
    program = ProgramModule.Program()
    with pytest.raises(NotImplementedError, match='requires multiprocessing fork'):
        program.run_scip(verbose=False)
    assert list(tmp_path.iterdir()) == []


def test_empty_product_is_one_in_constraint_with_parameters():
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.constraints = [[[], ['-1', 'objvar'], ['==']]]
    result = program.run_scip(verbose=False)
    for endpoint in result[:2]:
        assert endpoint['primal'] == pytest.approx(1)
        assert endpoint['dual'] == pytest.approx(1)


def test_simplification_retains_tiny_constant_relations_without_rescaling():
    program = ProgramModule.Program()
    constraint = [['1e-30'], ['==']]
    program.constraints = [constraint]
    program.optimize_remove_numeric_lines()
    assert program.constraints == [constraint]


def test_empirical_roundoff_is_evaluated_at_solver_tolerance():
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.constraints = [[['0.48500000000000004'], ['0.515'], ['-1.0'], ['==']],
                           [['objvar'], ['-0.25'], ['==']]]
    result = program.run_scip(verbose=False)
    for endpoint in result[:2]:
        assert endpoint['primal'] == pytest.approx(0.25)
        assert endpoint['dual'] == pytest.approx(0.25)


@pytest.mark.parametrize('value', [-2, 2])
@pytest.mark.parametrize('sense', ['min', 'max'])
def test_pip_roundtrip_preserves_objectives_outside_ate_range(tmp_path, value, sense):
    from pyscipopt import Model
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], [str(-value)], ['==']]]
    path = tmp_path / 'bounds.pip'
    program.to_pip(path, sense=sense)
    model = Model()
    model.hideOutput()
    model.readProblem(str(path))
    model.optimize()
    assert model.getPrimalbound() == pytest.approx(value)
    assert model.getDualbound() == pytest.approx(value)


def test_pip_export_preserves_caller_owned_stream():
    stream = io.StringIO()
    program = ProgramModule.Program()
    program.parameters = ['objvar']
    program.to_pip(stream)
    assert 'objvar free' in stream.getvalue()

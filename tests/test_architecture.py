"""Cross-layer contracts: immutable compilation, backend scope and ownership."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction
import ast
import importlib
from pathlib import Path
import pickle

import pytest
from autobounds import (DAG, Bounder, causalProblem, Program, LinearResponseModel,
                       MathematicalModel, Variable, Polynomial, Term, Constraint)
from autobounds.backends.scip import ScipBackend
from autobounds.backends.cdd import CddBackend
from autobounds.polynomial import simplify_model


def test_compilation_snapshots_mutable_legacy_rows():
    program = Program()
    program.parameters = ['x', 'objvar']
    program.constraints = [[['x'], ['-0.25'], ['==']],
                           [['x'], ['-1', 'objvar'], ['==']]]
    model = program.compile(limits=(-2, 3))
    program.constraints[0][1][0] = '-0.75'
    program.parameters.append('y')
    assert model.constraints[0].expression.constant == Fraction(-1, 4)
    assert len(model.variables) == 2
    assert model.variables[-1].lower == -2
    assert model.variables[-1].upper == 3
    with pytest.raises(FrozenInstanceError):
        model.variables[0].lower = -3
    simplified, fixed = simplify_model(model)
    assert fixed == {'x': Fraction(1, 4), 'objvar': Fraction(1, 4)}
    assert model.variables == simplified.variables
    assert model.constraints[1].expression.degree == 1


def test_exact_and_numeric_backends_use_the_same_response_model():
    pytest.importorskip('cdd.gmp')
    declaration = LinearResponseModel([[0, 1, 1]], [0, 0, 1], labels=['observed'])
    model = declaration.compile()
    assert CddBackend.supports(model)
    assert not ScipBackend.supports(model)
    symbolic = CddBackend().solve(model)
    assert symbolic.evaluate({'observed': Fraction(3, 5)}) == (0, 0.6)
    bound = model.bind({'p0': Fraction(3, 5)})
    assert ScipBackend.supports(bound)
    result = ScipBackend().solve(bound, verbose=False)
    assert result.lower.primal == pytest.approx(0)
    assert result.lower.dual == pytest.approx(0)
    assert result.upper.primal == pytest.approx(.6)
    assert result.upper.dual == pytest.approx(.6)
    assert model.parameters[0].label == 'observed'
    assert bound.parameters == ()
    with pytest.raises(ValueError, match='exactly'):
        model.bind({})
    with pytest.raises(ValueError, match='bind symbolic'):
        ScipBackend().solve(model, verbose=False)


def test_backend_domains_and_nonlinear_objective_are_explicit():
    model = MathematicalModel((Variable('x', -2, 3),), (),
                              Polynomial((Term(1, ('x', 'x')),)))
    assert not CddBackend.supports(model)
    with pytest.raises(ValueError, match='normalized linear response'):
        CddBackend().solve(model)
    result = ScipBackend().solve(model, verbose=False)
    assert result.lower.primal == pytest.approx(0, abs=1e-6)
    assert result.lower.dual == pytest.approx(0, abs=1e-6)
    assert result.upper.primal == pytest.approx(9, abs=1e-6)
    assert result.upper.dual == pytest.approx(9, abs=1e-6)


def test_ir_rejects_unknown_variables_and_duplicate_declarations():
    with pytest.raises(ValueError, match='Undeclared'):
        MathematicalModel((), (), Polynomial((Term(1, ('x',)),)))
    with pytest.raises(ValueError, match='unique'):
        MathematicalModel((Variable('x'), Variable('x')), (), Polynomial())
    with pytest.raises(ValueError, match='Unsupported'):
        Constraint(Polynomial(), '<')


@pytest.mark.parametrize('copy_problem', [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_workflow_ownership_survives_copy_and_serialization(copy_problem):
    problem = causalProblem(DAG('X -> Y'))
    problem.set_ate('X', 'Y')
    copied = copy_problem(problem)
    for name in ('continuous', 'inference', 'replay'):
        assert getattr(copied, f'_{name}_workflow').problem is copied
        assert getattr(problem, f'_{name}_workflow').problem is problem
    copied._operation_log.clear()
    assert problem._operation_log


def test_engine_and_workflows_do_not_import_orchestration_facade():
    root = Path(importlib.import_module('autobounds').__file__).parent
    paths = [root / 'Bounder.py', root / 'data_utils.py', root / 'model.py',
             *sorted((root / 'workflows').glob('*.py')),
             *sorted((root / 'backends').glob('*.py'))]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ('causalProblem', 'autobounds.causalProblem'), path


def test_failed_worker_creation_cleans_backend_owned_directory(monkeypatch, tmp_path):
    module = importlib.import_module('autobounds.backends.scip')
    class FailingContext:
        def Process(self, **kwargs):
            raise RuntimeError('cannot create worker')
    monkeypatch.setattr(module, 'get_context', lambda *args: FailingContext())
    monkeypatch.setattr(module.tempfile, 'tempdir', str(tmp_path))
    program = Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], ['==']]]
    with pytest.raises(RuntimeError, match='cannot create worker'):
        program.run_scip(verbose=False)
    assert list(tmp_path.iterdir()) == []


def test_spawn_reports_model_construction_error_and_cleans_workers(monkeypatch, tmp_path):
    import multiprocessing
    module = importlib.import_module('autobounds.backends.scip')
    monkeypatch.setattr(module.tempfile, 'tempdir', str(tmp_path))
    before = {child.pid for child in multiprocessing.active_children()}
    # This exact coefficient cannot be represented by SCIP's floating point API.
    model = MathematicalModel((Variable('x'),), (),
                              Polynomial((Term(10**1000, ('x',)),)))
    with pytest.raises(RuntimeError, match='OverflowError'):
        ScipBackend().solve(model, verbose=False)
    assert {child.pid for child in multiprocessing.active_children()} == before
    assert list(tmp_path.iterdir()) == []


def test_spawn_timeout_cleans_workers_before_removing_files(monkeypatch, tmp_path):
    import multiprocessing
    module = importlib.import_module('autobounds.backends.scip')
    monkeypatch.setattr(module.tempfile, 'tempdir', str(tmp_path))
    before = {child.pid for child in multiprocessing.active_children()}
    model = MathematicalModel((Variable('x'),), (), Polynomial((Term(1, ('x',)),)))
    result = ScipBackend().solve(model, verbose=False, maxtime=0, return_dgps=True)
    assert result.lower.end == result.upper.end == -1
    assert all(item['status'] != 'ok' for item in result.dgps.values())
    assert {child.pid for child in multiprocessing.active_children()} == before
    assert list(tmp_path.iterdir()) == []


def test_legacy_inspection_models_are_lazy_and_unsolved():
    program = Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], ['-0.4'], ['==']]]
    program.run_scip(verbose=False)
    assert program._scip_backend._inspection_models == {}
    assert program.M_lower.getObjectiveSense() == 'minimize'
    assert program.M_upper.getObjectiveSense() == 'maximize'
    assert program.M_lower.getNSols() == program.M_upper.getNSols() == 0
    assert program.last_result.lower.primal == pytest.approx(.4)


def test_general_model_export_and_solver_keep_one_sided_domains(tmp_path):
    from pyscipopt import Model
    from autobounds.exporters import write_pip
    model = MathematicalModel((Variable('x', None, 2),),
                              (Constraint(Polynomial((Term(1, ('x',)), Term(3))), '>='),),
                              Polynomial((Term(1, ('x',)),)))
    result = ScipBackend().solve(model, verbose=False)
    assert result.lower.primal == result.lower.dual == -3
    assert result.upper.primal == result.upper.dual == 2
    for sense, expected in [('min', -3), ('max', 2)]:
        path = tmp_path / f'{sense}.pip'
        write_pip(model, path, sense)
        compiled = Model()
        compiled.hideOutput()
        compiled.readProblem(str(path))
        compiled.optimize()
        assert compiled.getPrimalbound() == expected
        assert compiled.getDualbound() == expected


def test_legacy_results_are_copies_of_structured_solver_results():
    program = Program()
    program.parameters = ['objvar']
    program.constraints = [[['objvar'], ['-0.4'], ['==']]]
    legacy, dgps = program.run_scip(verbose=False, return_dgps=True)
    legacy[0]['primal'] = 99
    dgps['lower']['values']['objvar'] = 99
    assert program.last_result.lower.primal == pytest.approx(.4)
    assert program.last_result.dgps['lower']['values']['objvar'] == pytest.approx(.4)
    program.simplify()
    assert program.last_result is None


def test_both_public_problem_builders_expose_shared_compilation():
    from autobounds import SymbolicProblem
    graph = DAG('X -> Y, U -> X, U -> Y', unob='U')
    numeric, symbolic = Bounder(graph), SymbolicProblem(graph)
    numeric.set_ate('X', 'Y')
    symbolic.set_ate('X', 'Y')
    assert isinstance(numeric.compile(), MathematicalModel)
    assert isinstance(symbolic.compile(), MathematicalModel)
    assert ScipBackend.supports(numeric.compile())
    assert CddBackend.supports(symbolic.compile())


def test_workflow_child_factory_preserves_legacy_subclass_behavior():
    class CustomProblem(causalProblem):
        def __init__(self, dag, *, required):
            super().__init__(dag)
            self.required = required
    parent = CustomProblem(DAG('X -> Y'), required='custom')
    child = parent._new_child_problem(DAG('X -> Y'), {})
    assert type(child) is causalProblem
    assert child._continuous_workflow.problem is child

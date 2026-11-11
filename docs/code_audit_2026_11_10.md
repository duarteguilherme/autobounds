# Code and repository audit — 10 November 2026

The subsequent [transformation audit](simplifications.md) removes the unsafe
`simplify_linear` path and remaining dormant Couenne helpers, and documents
the conservative replacement and additional expression/export fixes.

## Follow-up fixes

The concrete defects below have now been addressed:

- Constant constraints use exact rational arithmetic on their decimal tokens.
  True relations are removed; false relations are retained unchanged for SCIP
  to evaluate with its normal feasibility tolerance. This preserves genuine
  contradictions without amplifying empirical floating-point roundoff. Empty products equal one. Both direct
  programs and zero-parameter substitutions have solver regression tests.
- Pyomo/Couenne execution implementations are removed. Compatibility methods
  raise `NotImplementedError` with a SCIP migration path. Unimplemented exports,
  subtraction bounds, canonical index lookup, and helpers for the removed
  `get_expr` API now fail explicitly. The historical log reader no longer asks
  for terminal input.
- Cardinality constructors use `None` defaults and the canonical model owns a
  copy of its configuration. Caller mutation and cross-model isolation are
  covered for Bounder, causalProblem, Parser, closedProblem and canonicalModel.
- Ineffective constraint comparisons now test identified observed probabilities
  and ATE endpoints on two graphs, checking primal and dual outputs. A previous
  test expected bounds from probabilities summing to 1.95; it now correctly
  checks infeasibility. Parameter substitution tests also verify their input
  remains unchanged.
- Spawn-only platforms fail explicitly before solver models or temporary files
  are created. This does not implement Windows/spawn support.

Sections below retain the original audit findings as historical context;
items 1–3 and the ineffective assertions in item 4 are resolved by this
follow-up. CI/version-matrix work, dependency pinning, large-asset migration,
and broader refactoring remain maintenance work, not claims of completed
validation. Historical scientific workflows remain archived and unvalidated.

### Follow-up validation

The source suite passes **270 tests** (18.06 seconds), including regressions
for numerical roundoff as well as infeasibility. The numerical CI example
again returns matching primal/dual point bounds of approximately
`[-0.142328, 0.442498]` and finite intervals. The symbolic factorial example
completes. Source and wheel distributions build successfully. The same **270
tests pass against the installed wheel** outside the checkout (18.34 seconds),
using the existing environment dependencies.

## Scope and method

Reviewed package sources, packaging configuration, all ten tracked example
scripts, ten demo notebooks, test collection, Docker instructions, and the
replication directory. Ran the test suite, built the wheel and source archive,
and checked examples. Notebook checks ran transformed Python cells in a
scratch copy, skipping installation commands and stopping before the first
solver call. They establish early failures, not end-to-end notebook validity.
The frozen replication implementation was inspected but not executed as a
scientific replication. No dependency vulnerability scan or Windows/macOS run
was performed.

Environment: Linux, Python 3.13, NumPy 2.3.5, SymPy 1.14, PySCIPOpt 6,
pycddlib 3.0.2. Dependency versions are observations, not a supported-version
matrix. No new QE/Redlog backend was implemented in this cleanup.

## Fixed or removed in this audit

| Finding | Action |
| --- | --- |
| `autobounds/ID.py` has an empty method body and raises `IndentationError`; the initial wheel shipped it | Removed the unimplemented, unreferenced module; added package-source compilation test |
| Extensionless `autobounds/ProgramUtils` duplicates an obsolete implementation | Removed; retained the imported `ProgramUtils.py` |
| `DAG.plot` is defined twice | Removed the overwritten definition |
| `SCM.draw_sample` splits multi-character variable names into letters | Preserved topological-order names; added deterministic sampling regression |
| `Program.get_bounds_scip()` calls native SCIP on parent models never optimized by the worker processes | Cached worker-reported primal/dual results; getter returns copies and rejects unavailable results. The pre-fix reproducer crashed with a native SCIP error/segmentation fault in this environment |
| `test_load_data` appears three times in one module, hiding two definitions | Removed the identical duplicate and restored one interventional-data test; checked both primal and dual endpoints; added duplicate-test-name guard |
| Tests import checkout-only `autobounds.autobounds` | Migrated to installed `autobounds` imports and removed the empty root package marker |
| Two paper example entry points have inconsistent sibling imports | Both direct-script and module launch modes now work; covered by subprocess regression tests |
| Direct `statsmodels` use relies on a transitive dependency | Added explicit dependency; added `dev` extra |
| Package discovery could include unrelated directories | Restricted discovery to `autobounds` and subpackages; restricted default pytest discovery to `tests` |
| Tracked bytecode, notebook checkpoints, macOS metadata, editor backups, solver logs | Removed generated files; expanded ignore rules |
| Empty/commented test modules and a broken historical analysis script | Removed empty test modules; archived the analysis script |

Existing public compatibility facades (`bounder.py`, `causalProblem`, `Query`)
remain. Their continued use is not evidence that they can safely be deleted.

## Historical demonstrations and installation material

Moved the old notebook collection and data to `archive/legacy_demos/` and the
Docker recipe/source archive to `archive/docker/`. This preserves provenance;
it does not reduce Git history size or repair historical relative paths.
Published replication sources and substantive data/results remain in place.

| Notebook, relative to former `autobounds_demo/` | Observed result |
| --- | --- |
| `apsa2023/iv_demo.ipynb` | Missing `autobounds.Query` module |
| `apsa2023/klm_demo.ipynb` | Missing `autobounds.Query` module |
| `apsa2023/outcome_selection_demo.ipynb` | Missing `autobounds.Query` module |
| `iv_demo.ipynb` | Missing `autobounds.Query` module |
| `klm_demo_relaxassumptions.ipynb` | Missing `autobounds.Query` module |
| `measurement_error_demo.ipynb` | Missing `Bounder.query` method |
| `mediation_demo.ipynb` | Missing `Bounder.query` method |
| `iv_more_categories_demo.ipynb` | Reached first solver; later cells not validated |
| `outcome_selection_demo.ipynb` | Reached first solver; later cells not validated |
| `nb_template.ipynb` | Non-solver template cells completed |

Additional static findings include obsolete `run_scip(filename=...)` calls,
local CSV paths that do not match the checkout, and unsupported Couenne flows.
The old Docker recipe uses Ubuntu 20.04/Python 3.8 (below the package's Python
3.10 minimum) and expects an externally supplied repository ZIP. It was not
rebuilt. Root installation instructions now point to source installation.

## Remaining issues, in recommended order

### 1. High: preserve infeasibility during simplification

`Program.optimize_remove_numeric_lines()` drops all numeric-only constraints,
regardless of whether they are true. Minimal reproducer:

```python
from autobounds import Program
program = Program()
program.constraints = [[['1'], ['==']]]  # 1 == 0, impossible
program.optimize_remove_numeric_lines()
assert program.constraints == []        # Current, incorrect simplification
```

`Bounder.write_program()` invokes this method, so the issue can affect models
after zero-parameter substitutions. Fixing it requires a defined policy for
constant arithmetic and tolerance, including empty products, signed constants,
equalities and inequalities. Preserve contradictory constraints for the solver
or report infeasibility explicitly. Add end-to-end primal/dual regression
coverage before changing the simplifier. This cleanup does not fix this issue.

### 2. Medium: retire or repair unsupported solver/API branches

`run_pyomo` solves models again after its first solve pass; its multiprocessing
lambda is unsuitable for portable multiprocessing. Couenne's
`ProgramUtils.get_final_bound` contains an interactive `input()` call.
`closedProblem.get_subtraction_bounds` also prints diagnostics and blocks on
`input()`. `Program.to_obj_pyomo` and `Program.to_cip` silently return `None`
from placeholder bodies. Helpers in `utils.py` expect old `get_expr` and
`.program` interfaces. These are source findings, not successful backend runs.

Prefer a documented deprecation/removal cycle or explicit unsupported errors,
then retain only backends with automated integration coverage. Do not advertise
Couenne/Pyomo support based on SCIP tests. The numerical process implementation
also assumes fork; spawn-only platforms remain unvalidated.

### 3. Medium: mutable configuration ownership

Several constructors use mutable dictionary defaults;
`canonicalModel.from_dag` retains and fills the caller's `number_values`
dictionary. Normalize defaults to `None` and copy configuration at ownership
boundaries. Add tests creating multiple models with differing cardinalities and
tests verifying caller inputs are unchanged. Avoid a broad mechanical rewrite
until intentional sharing in replay/caching is distinguished from accidental
sharing.

### 4. Medium: improve assertions and reproducible verification

Two expressions in `tests/test_bounder.py::test_load_data` compare constraints
without asserting anything. Some older tests merely run or print results.
Replace stale expected representations with mathematically meaningful checks;
automatically inserting `assert` can preserve outdated expectations. Add a CI
matrix for supported Python versions and an installed-wheel job. Package-source
compilation and duplicate-test-name checks now guard two previous blind spots.
Full coverage/performance thresholds and statistical Monte Carlo validation
remain separate work.

### 5. Low: repository maintenance and dependency policy

The archive still contains large historical data and the SCIP tarball. Moving
them preserves access but does not shrink clones. A future release could place
reproducibility assets in a separately versioned archive, with provenance links.
The frozen replication package includes its own unfinished `ID.py`; it is not
installed or covered by active-package compilation checks. Do not mass-format
or rewrite it as if it were current source.

Core modules still mix CRLF/LF, wildcard imports, broad exception handlers,
long functions and stale docstrings. Refactor one responsibility at a time,
starting with solver lifecycle and constraint representation. No formatter or
linter executable was available during this audit; syntax checks and manual
review are not a substitute for a lint baseline. Dependencies are broadly
unpinned; define a tested compatibility range before claiming reproducibility
across future dependency releases.

## Verification

- Full source-tree pytest suite: **238 passed in 17.50 seconds**.
- Source archive and wheel built with `python -m build --no-isolation`.
- All nine CLI examples accept `--help`; the tenth (`symbolic_factorial.py`)
  completed its exact symbolic solve. Both paper scripts also pass module-mode
  startup tests.
- Numerical smoke run:
  `python examples/iv_ate_ci_subsampling.py --n 200 --nsamples 3 --ci-workers 1`.
  Both primal and dual point endpoints were approximately `[-0.142328, 0.442498]`;
  95% and 98% intervals were produced. Three replications are a functionality
  check, not evidence of nominal coverage.
- Installed-wheel verification: **238 passed in 18.01 seconds**, running copied
  tests/examples outside the checkout with imports verified under the wheel
  installation directory. Wheel contents contain only the active package and
  distribution metadata (21 files), with no unfinished `ID.py`. This reuses the
  environment's dependencies; it is not a clean dependency-resolution experiment.

The audit identifies concrete defects and establishes a cleaner supported
surface. Passing the checks does not establish correctness of every model,
backend, historical notebook, or statistical procedure.

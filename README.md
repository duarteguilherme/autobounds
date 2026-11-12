# Autobounds

Autobounds is an advanced software tool designed to automate the calculation of partial identification bounds in causal inference problems (we're still in beta, so expect many changes soon).

Developed by researchers at the University of Pennsylvania, Johns Hopkins University, and Princeton University, Autobounds leverages polynomial programming and dual relaxation techniques to compute the sharpest possible bounds on causal effects, even when data is incomplete or mismeasured. You can learn more in their research paper.

This tool is particularly valuable in fields like economics, political science, and epidemiology, where researchers often face challenges such as confounding, selection bias, measurement error, and noncompliance. By automating the process, Autobounds allows for more precise and reliable estimation of causal relationships, facilitating better-informed decision-making.

Install from source with Python 3.10 or newer:

```bash
python -m pip install -e .
```

See [examples](examples/README.md) for current entry points. Historical notebooks
and the old Docker recipe are preserved in [archive](archive/README.md); they
are not supported setup instructions.
The [JASA paper examples](docs/paper_examples.md) reproduce the main IV,
selection, measurement-error, and missing-data applications using the current API.
See the [code audit](docs/code_audit_2026_11_10.md) for validation scope and
the status of fixes and remaining maintenance work.

Development is currently being conducted by Guilherme Duarte, Dean Knox, and Kai Cooper. Code contributions were also made by Lisa Schulze-Bergkamen and Jeremy Zucker.

## API structure

See [architecture and backend contracts](docs/architecture.md) for compilation,
model ownership, supported backend capabilities, and compatibility boundaries.

- `Bounder`: single bounds problem engine (one model/estimand/data solve flow).
- `SymbolicProblem`: exact symbolic bounds for supported linear response models.
- `causalProblem`: orchestrator that can manage one or more bounders.
  - For backward compatibility, single-problem calls on `causalProblem` still work via an implicit default bounder.
  - New code should prefer explicit `Bounder` usage when solving a single problem.

## Traditional numerical workflow

Define a causal graph, load observed probabilities, and optimize a causal
estimand. Here X and Y are binary, and U is an unobserved common cause:

```python
import pandas as pd
from autobounds import DAG, causalProblem

dag = DAG("X -> Y, U -> X, U -> Y", unob="U")
problem = causalProblem(dag)
problem.load_data(pd.DataFrame({
    "X": [0, 0, 1, 1],
    "Y": [0, 1, 0, 1],
    "prob": [0.3, 0.2, 0.1, 0.4],  # Joint probabilities P(X,Y).
}))
problem.add_prob_constraints()
problem.set_ate("X", "Y")  # E[Y(X=1)] - E[Y(X=0)].
program = problem.write_program()
result = program.run_scip(verbose=False)
print(result[0]["dual"], result[1]["dual"])  # Approximately -0.3, 0.7.
```

SCIP is the supported numerical backend and requires `pyscipopt`.
The historical Pyomo and Couenne backends are retired; their entry points now
raise `NotImplementedError` directing callers to `run_scip()`. Numerical solver
workers use multiprocessing `spawn` and construct SCIP models inside each worker;
they no longer require `fork`. When running a Python script, put code that starts
solves behind a main guard (including calls to `problem.solve()`):

```python
def main():
    # Build the problem and run the solve here.
    ...

if __name__ == "__main__":
    main()
```

Workers start fresh interpreters. Public interfaces and plotting/statistical
dependencies load on demand to keep numerical worker startup small, but some
startup cost remains. The `maxtime` wall-clock budget includes worker startup. Removal of the
fork requirement does not itself constitute verification on Windows or macOS.

Constant tautologies are recognized exactly from their decimal tokens. False
relations are retained for SCIP to evaluate with its normal feasibility tolerance. For new single-problem
code, `Bounder` can replace `causalProblem` in this example.

## Polynomial simplification

Use `program.simplify()` before `run_scip()` to collect repeated terms, remove
exact duplicate constraints, and propagate fixed-variable equalities while
retaining their equations and domains. It returns a reduction report and is
opt-in. The old `simplify_linear()` method has been removed.
See [supported simplifications and audit findings](docs/simplifications.md).

## Symbolic linear bounds

For supported linear models, `SymbolicProblem` returns exact bound formulas
in the observed probabilities. Declare which distributions are available, set
the estimand, and call `solve()`. The lower bound is a maximum of affine
expressions in those probabilities; the upper bound is a minimum.

Install the optional exact polyhedral backend (the source build requires
cddlib and GMP development headers):

```bash
python -m pip install -e '.[symbolic]'
```

Declare the available distributions without supplying numerical probabilities:

```python
from autobounds import DAG, SymbolicProblem

dag = DAG("Z -> X, X -> Y, U -> X, U -> Y", unob="U")
problem = SymbolicProblem(dag)
problem.load_data("X,Y", cond="Z")  # Every cell of P(X,Y | Z=z).
problem.set_estimand(problem.p("Y(X=1)=1") - problem.p("Y(X=0)=1"))

model = problem.write_program()  # Explicit M, c, and symbolic data labels.
bounds = problem.solve()
print(bounds.lower)  # SymPy Max of affine expressions in the data probabilities.
print(bounds.upper)  # SymPy Min of affine expressions in the data probabilities.
```

`load_data("X,Y")` declares a joint observational distribution;
`load_data("Y", do="X=0")` declares an experimental distribution. Declarations
accumulate, so observational and experimental data can be combined. Use
`add_data(problem.p("Y(X=1)=1"), label="trial")` for a single named probability.
`set_p_to_zero(problem.p("X(Z=0)=1&X(Z=1)=0"))` excludes IV defiers, imposing
monotonicity. Cardinalities are supplied as in `Bounder`, for example
`SymbolicProblem(dag, number_values={"Z": 3})`.

The graph adapter currently requires data and estimands to involve one
unrestricted response component. Each nonsingleton component must have a latent
parent shared by all its variables. Conditioning is supported on independent
observed roots, where conditional cells can be represented by intervention
events; these conditional distributions must exist. Nonlinear queries,
conditional ratios, and declarations spanning multiple independent components
are rejected. In particular, the adapter does not silently pool independent
latent variables. For other already-derived linear models, construct
`LinearResponseModel(matrix, target, labels=...)` directly: its model is
`q >= 0`, `sum(q) = 1`, `matrix @ q = p`, with target `target @ q`.

The exact GMP/cdd solver computes the convex hull of the response points
`(matrix[:, s], target[s])` using pycddlib's exact GMP rational arithmetic.
It removes dependent observable coordinates, converts the hull to inequalities,
and extracts both sharp endpoint envelopes. Identified targets are recovered
from their affine relation to the data. Every bound certificate is checked
exactly against all response types; the backend never falls back to floats.

The complete sharp envelopes include negative and fractional coefficients and
support signed estimands such as ATEs. `lower_certificates` and
`upper_certificates` give coefficients multiplying `(1, *model.symbols)`;
`compatibility_equalities` records data identities, and
`compatibility_inequalities` contains target-free hull inequalities. These
conditions, together with `bounds.lower <= bounds.upper`, describe the
observable compatibility region.
Sharpness assumes population data compatible with the model. Optional
`bounds.evaluate({label: probability, ...})` substitutes numerical values but
does not check compatibility. Expressions may be redundant on the compatibility
region; completeness does not imply an irredundant facet list.

The solver returns complete sharp envelopes. Hull enumeration can still be
expensive on models with many facets; a failed conversion raises an error
instead of returning a partial family labeled complete.

For the binary factorial example, combine observational and experimental data:

```python
from itertools import product

dag = DAG("A -> Y, B -> Y, U -> A, U -> B, U -> Y", unob="U")
problem = SymbolicProblem(dag)
problem.load_data("A,B,Y")
for a, b in product(range(2), repeat=2):
    problem.load_data("Y", do=f"A={a},B={b}")
problem.set_estimand(problem.p("Y(A=1)=1"))  # Y_1B, with B natural.
bounds = problem.solve()
```

This graph assumes that intervening on A does not change B. The shared latent
parent permits an unrestricted distribution over the 64 response types.
Additional observational margins are already implied by the joint distribution.
A runnable example printing the complete formulas and solve time is available
in [examples/symbolic_factorial.py](examples/symbolic_factorial.py).

## Confidence Intervals (Subsampling)

`causalProblem.solve(ci=True, ...)` now computes confidence intervals using subsampling (without replacement) by replaying your loaded datasets and model steps.
Prefer loading analysis data with `read_data(...)` for this flow (legacy `load_data(...)` is still supported).

Default subsample size per dataset:

- `m = n` if `n < 120`
- otherwise `m = min(n, max(80, floor(n**0.7)))`

Important options:

- `nsamples` (default `200`): number of subsampling replications
- `subsample_rate` (default `0.7`): exponent used when `subsample_size` is not set
- `subsample_size` (default `None`): explicit fixed `m` per dataset (overrides rate rule)
- `ci_method`:
  - `"empirical_subsample_quantile"` (legacy default): direct quantiles of subsample bounds
  - `"recentered_subsampling"`: uses `sqrt(b) * (theta_b - theta_n)` recentering and `1/sqrt(n)` back-transform

Example:

```python
result = problem.solve(
    ci=True,
    nsamples=300,
    subsample_rate=0.7,
    ci_method="recentered_subsampling",
    verbose_optimizer=False,
    verbose_result=False,
)
```

## Install from source

Use a recent Python (>=3.10) and a clean virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

This installs Autobounds in editable mode for development. For a regular install, replace the last line with:

```bash
python -m pip install .
```

## Optional features

The default install includes numerical solving, causal declarations, empirical
subsampling, and the existing SymPy-based closed-form APIs. It does not install
plotting libraries or regression/inference libraries.

| Extra | Features | Additional dependencies |
| --- | --- | --- |
| `plot` | DAG and bound plots | matplotlib, plotnine |
| `inference` | Regression-based covariate models, Gaussian/KL data constraints | scipy, statsmodels |
| `symbolic` | Exact cdd hull bounds | pycddlib, with native cddlib/GMP prerequisites |

For a checkout, install the extras you need:

```bash
python -m pip install '.[plot]'
python -m pip install '.[inference]'
python -m pip install '.[plot,inference,symbolic]'
```

Use `autobounds[plot]` (and analogous combinations) when installing a published
release. Existing installations keep their already-installed dependencies;
fresh installs must request extras for these features. Missing-feature errors
include the relevant install command. Plotnine has its own statistical
dependencies, so `plot` also brings some inference libraries transitively.
The `dev` extra provides pytest and build tools; combine it with all feature
extras to run the complete suite without optional-dependency skips.

## Continuous integration

[Installed wheel tests](https://github.com/duarteguilherme/autobounds/actions/workflows/tests.yml)
build and install the package in fresh environments on Linux, macOS, and Windows
with Python 3.10 and 3.13. Separate Linux jobs test each extra alone and all
three feature extras together.
See [clean-install CI](docs/ci.md) for coverage and local reproduction commands.

## Build and publish (PyPI)

See the [release checklist](docs/releasing.md), [changelog](CHANGELOG.md), and
[0.1 migration guide](docs/migration-0.1.md) before preparing a release.

Build source and wheel distributions:

```bash
python -m pip install build twine
python -m build
```

Optional: upload to TestPyPI to verify packaging (requires an API token):

```bash
python -m twine upload -r testpypi dist/*
```

Publish to PyPI (requires a PyPI token):

```bash
python -m twine upload dist/*
```

### Direct symbolic linear equalities

`add_constraint(expression, label="t")` declares `expression = t`, where `t`
is a symbolic right-hand side. Unlike event data, the expression may be a
signed rational contrast of queries:

```python
n0 = problem.p('Y(X=1)=1') - problem.p('Y(X=0)=1')
n1 = problem.p('Y(X=0)=1')
problem.set_estimand(n0)
first = problem.solve()
problem.add_constraint(n0, label="t")
problem.set_estimand(n1)
second = problem.solve()
t = next(symbol for symbol in second.symbols if str(symbol) == "t")
sequential_lower = second.lower.subs(t, first.lower)
```

Here `problem` must use a graph containing `X,Y` and supported by the symbolic
adapter. Substitute `t` in the compatibility conditions as well as the bounds;
formulas describe feasible distributions only on that compatibility region.
A maximum of several first-stage expressions can be treated one active piece
at a time. The label is a name, not a parsed formula or a numeric right-hand
side. Substitute numeric values after solving.

This is lexicographic optimization. It certifies simultaneous minimizers only
if the second-stage minimum equals the unrestricted minimum of `n1`; it does
not generally minimize a weighted sum. Repeat with maxima for upper bounds.
The fixed-coefficient interface supports equalities with rational coefficients, not
inequalities, nonlinear constraints, or symbolic objective coefficients.
At the matrix level, `LinearResponseModel(..., constraints=[row],
constraint_labels=["t"])` provides the same functionality. Exact certificates
refer to the concatenated data and constraint rows.


### Automatic parametric LP reduction

For models with observed independent roots and homogeneous effects across
response strata, use `SymbolicProblem(...)`; solver selection is automatic. Declare the
graph, symbolic data, zero events, and `add_homogeneity` assumptions;
AutoBounds discovers the response features, derives a parametric LP when
possible, and eliminates nuisance coordinates symbolically. The generated
feature map, LP, and reduction report are inspectable. Unsupported
nonlinearities or unproved coefficient signs raise errors.

See [symbolic reduction details](docs/simplifications.md).

Simple supported models retain the fast linear-response solver. Models needing
reduction select the automatic reduction path before solving. Inspect
`problem.solver_selection` for the chosen method and reason. Unexpected solver
errors propagate with their traceback; they do not trigger a silent fallback.
Unsupported models raise `UnsupportedModelError` with the unsupported feature
and any applicable next step. This indicates a solver capability limit, not
infeasibility of the causal model. Invalid data and projection-budget limits
raise separate `ValueError` messages. `parametric=True` or `False` can
still force a method for debugging, but ordinary declarations need no flag.

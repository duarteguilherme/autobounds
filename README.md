# Autobounds

Autobounds is an advanced software tool designed to automate the calculation of partial identification bounds in causal inference problems (we're still in beta, so expect many changes soon).

Developed by researchers at the University of Pennsylvania, Johns Hopkins University, and Princeton University, Autobounds leverages polynomial programming and dual relaxation techniques to compute the sharpest possible bounds on causal effects, even when data is incomplete or mismeasured. You can learn more in their research paper.

This tool is particularly valuable in fields like economics, political science, and epidemiology, where researchers often face challenges such as confounding, selection bias, measurement error, and noncompliance. By automating the process, Autobounds allows for more precise and reliable estimation of causal relationships, facilitating better-informed decision-making.

To get started, download the software via Docker and run the following command to launch Autobounds:

```bash
docker run -p 8888:8888 -it gjardimduarte/autolab:v5
```

This will allow you to easily integrate Autobounds into your causal inference workflows.

To install it in your machine, clone this repo and use python -m pip install .

Development is currently being conducted by Guilherme Duarte, Dean Knox, and Kai Cooper. Code contributions were also made by Lisa Schulze-Bergkamen and Jeremy Zucker.

## API structure

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

This numerical workflow requires SCIP and `pyscipopt`. For new single-problem
code, `Bounder` can replace `causalProblem` in this example.

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

## Build and publish (PyPI)

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

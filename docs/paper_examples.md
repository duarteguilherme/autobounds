# Examples from the AutoBounds JASA paper

The maintained runner is [examples/paper_discrete.py](../examples/paper_discrete.py).
It implements the main population-bound applications in Sections 7.1 and 7.3
of Duarte, Finkelstein, Knox, Mummolo, and Shpitser (2024),
[*An Automated Approach to Causal Inference in Discrete Settings*](https://doi.org/10.1080/01621459.2023.2216909),
JASA 119(547), 1778–1793 ([open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11566246/)).

## Run the examples

From the repository root, install the current Python package, then run:

```bash
python -m pip install -e .
python examples/paper_discrete.py --example iv_standard_ate
python examples/paper_discrete.py --example all --time-limit 120 \
    --output examples/outputs/paper_discrete.json
```

Only the base dependencies are required. SCIP is called through PySCIPOpt;
Couenne, Pyomo, R, and the old replication environment are not used. The module
also supports `python -m examples.paper_discrete` from the root. Dataset paths
are relative to the script, so an absolute script path works from other working
directories after installing AutoBounds.

Each line of output is a JSON result with separate lower/upper primal and dual
bounds. `converged` means both endpoint gaps meet the requested absolute `--gap`
(default `1e-5`), subject to numerical solver tolerances. `infeasible` is an
expected result for the incorrect monotonicity scenarios. `unfinished` means
the budget did not establish convergence; report the outer interval
`[lower.dual, upper.dual]`, not the inner primal interval, as the conservative
result. Missing/unbounded numeric fields are absent or `null`, not valid finite
endpoints. `--time-limit` applies to the solve for each example; compilation and
process startup add overhead.

## Map to the paper

| CLI name | Paper setting | Target | Reference result |
| --- | --- | --- | --- |
| `iv_cautious_ate` | Figure 5(a), allow Z → Y | ATE of X on Y | approximately [-.63, .37] |
| `iv_standard_ate` | Figure 5(b), exclusion without monotonicity | ATE | [-.5502, -.1460], Table 1 |
| `iv_monotone_ate` | Figure 5(c), incorrectly exclude defiers | ATE | infeasible |
| `iv_cautious_late` | Figure 5(a) | effect among compliers | [-1, 1] |
| `iv_standard_late` | Figure 5(b) | effect among compliers | [-1, 1] |
| `iv_monotone_late` | Figure 5(c) | effect among compliers | infeasible |
| `outcome_selection` | Figures 7(a), 8(a) | ATE | [-.50, .64] |
| `measurement_error` | Figures 7(b), 8(b) | ATE | [-.62, 1.00] |
| `nonresponse` | Figures 7(c), 8(c) | ATE | [-.25, .75] |
| `joint_missingness` | Figures 7(d), 8(d) | ATE | point identification near -.25 |

These are bounds, not the simulated true effects: the paper's IV example has
true ATE -.25 and true LATE -.36, neither of which should be substituted for the
identified interval. The LATE denominator is the probability of
`X(Z=1)=1 & X(Z=0)=0`; the current API imposes its documented positive
`safe_min` to make that conditional estimand well-defined.

## Graphs and assumptions

- IV: `Z -> X -> Y`, with `U -> X,Y`. The cautious graph also includes `Z -> Y`.
  The monotone graph excludes `X(Z=0)=1 & X(Z=1)=0` using `add_assumption`.
- Selection: `X -> Y -> S`, with `U -> X,Y`. Observed S=1 cells are supplied as
  **joint** probabilities; their sum is the known selection rate.
- Measurement: `X -> Y -> S`, with `U -> Y,S`. Here Y is the true outcome,
  S the proxy, and `S(Y=0)=1 & S(Y=1)=0` is excluded.
- Nonresponse: `X -> Y`, `X -> R`, `Y -> R`. Supply P(X,Y|R=1) and P(X,R).
- Joint missingness: `X -> Y`, `A -> B`, `Y -> B`, where A and B indicate
  observation of X and Y. The graph makes A independent of X, allowing P(X)
  to be recovered from treatment respondents. Observed selected cells are
  otherwise kept as joint probabilities.

The [data directory](../examples/paper_discrete_data/README.md) documents every
input table. Files are unchanged copies of `replication_files/data/`, whose
scripts and bundled historical package remain intact. No input is regenerated,
overwritten, or silently normalized. Finite CSV precision and rounded published
endpoints explain small differences from the printed values.

## Validation and scope

`tests/test_paper_discrete_examples.py` checks both primal and dual endpoints
against the references, checks the two infeasibility results, and tests the CLI
outside the checkout. Installed-wheel CI copies these examples and their data
into the isolated test environment.

Joint missingness can take substantially longer to certify than to find a
feasible value. Its test compares against an independent shadow-variable
identification calculation and checks feasible witnesses and outer bounds under
a short budget; it does **not** treat a timeout as a proof of sharpness. Increase
`--time-limit` for a tighter certificate. Solver runtimes and optimization
trajectories will differ from the paper's historical environment.

This runner covers the main population examples, not the 1,000-replication
coverage study in Section 7.2, appendix mediation/cardinality experiments, or
pixel-for-pixel recreation of the original figures. The historical versions of
those experiments remain in `replication_files/`.

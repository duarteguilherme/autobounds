# Supported simplifications and transformation audit

`Program.simplify_linear()` and its string/SymPy replacement helpers have been
removed. They were not covered by active solver tests, recognized only a narrow
subset of linear equalities, and could discard defining equations after
substitution. A linear-looking expression is not sufficient evidence that an
elimination preserves the feasible set.

## Using the replacement

### Automatic zero support reduction

`Program.compile()` now removes decision variables that the constraints prove
are zero. Numerical solves and PIP export use this reduced model automatically;
direct `ScipBackend.solve(model)` calls apply the same pass.

For nonnegative stratum probabilities, `2*q1 + 3*q2 == 0` or
`q1 + q2 <= 0` implies both strata are absent. The compiler substitutes zero
through all constraints and the objective, removes tautologies, and repeats
until no more zeros follow. It also recognizes powers and products whose other
factors are provably nonzero from their bounds or scalar equalities.

This is a conservative algebraic test, not a complete classifier of individual
and population assumptions. In particular:

- `q1 - q2 == 0` permits positive mass on both strata.
- `q1*q2 == 0` does not identify which independent factor is zero.
- `q1 + q2 <= theta` with any positive constant `theta` does not eliminate either
  stratum, even when that constant is very small. Detection uses exact arithmetic.
- Contradictions survive elimination, including conflicts with observed root
  probabilities and original variable domains.

Inspect `model.eliminated_values` for the removed names and exact zeros.
Successful SCIP DGP results restore those entries as zero. Compilation leaves
the `Program` declarations unchanged, so changing a zero constraint to a positive
tolerance and recompiling restores its support. Use
`program.compile(eliminate_zeros=False)` to inspect the unreduced Program;
this cannot undo exclusions already explicitly made in a causal declaration.
Similarly, changing an assumption after calling the in-place `simplify()` requires
retaining the original declarations.

A binary IV regression with uniform observed `(Z,X,Y)` probabilities and
`P(X(0)=1, X(1)=0) <= 0` removes four defier response types: the compiled
program shrinks from 17 variables to 13, including the objective variable.
Both primal and dual ATE endpoints remain `[-0.5, 0.5]`. Smaller support can
reduce polynomial size and solver work; this test does not establish a runtime
speedup, and SCIP may itself discover some of the same reductions.

### Optional additional simplification

```python
program = problem.write_program()
report = program.simplify()
print(report)
result = program.run_scip(verbose=False)
```

The method compiles an immutable `MathematicalModel`, applies
`polynomial.simplify_model()`, then updates the compatibility rows in place.
It returns before/after constraint and
term counts and the names of fixed and eliminated parameters. This additional
pass is opt-in; zero support reduction during compilation is automatic.

The supported transformations are:

- Multiply numeric factors and combine identical monomials with exact rational
  arithmetic on decimal tokens; preserve repeated variables as powers.
- Remove zero terms, true constant relations, and exact duplicate rows.
- Propagate equalities that fix one variable to a constant, including chains.
  Keep the defining equations and every variable's domain. Contradictory
  equations and fixed values outside [0, 1] therefore remain infeasible.
- Skip fixed values with nonterminating decimal expansions rather than round
  new coefficients. General linear elimination, division by variables, and
  treating inequalities as equalities are not performed.

These rewrites preserve the exact polynomial model represented by the input
coefficients. SCIP still solves numerically: degenerate constraints and
coefficients near its feasibility tolerance can behave differently after
algebraic rewriting. There is no promise of a speedup on every model; SCIP also
has its own presolve. Nonzero fixed variables retain their defining equations
and bounds; proven zero variables use the reconstruction mapping described above.

A tested observational model `X -> Y` with monotonicity, joint probabilities
`[.4, .1, .2, .3]` over `(X,Y) = (0,0),(0,1),(1,0),(1,1)`, and ATE objective
shrinks from **6 constraints / 16 terms to 4 constraints / 8 terms**. Both
primal and dual ATE endpoints stay at `0.4` before and after simplification.
This demonstrates model reduction, not a measured wall-clock speedup.

## Focused audit results

| Transformation or dormant path | Action |
| --- | --- |
| `simplify_linear`, `replace_linear`, `replace_expr_lin*`, narrow `is_linear` | Removed, along with commented-out prototype tests |
| Retired Couenne monitoring, process helpers, Pyomo product helper | Removed unused implementation and old half-second sleep path; public retired backend methods still raise migration errors |
| Explicit parameter substitution | Corrected repeated factors (`x*x` must become `value**2`); `optimize_add_param_value` now retains the fixed variable's equation and domain |
| First-node substitution | Copies the term's factor list instead of mutating caller-owned expressions |
| `Q.clean_list` | Removes unit factors before combining terms, so `x - 1*x` cancels in one pass |
| PIP export | Multiplies numeric factors; `2*3*x` exports as `6 x`, not `23 x`; handles empty and zero products, keeps the objective free instead of imposing `[-1, 1]`, and leaves caller-owned streams open |

Tests cover exact polynomial evaluation with a fixed random seed, fixed-variable
chains and powers, conflicting equations, out-of-domain fixed values,
inequalities, idempotence, primal/dual endpoint equivalence, and a causal-model
integration case. PIP files with objectives outside the ATE range are read back
into SCIP and checked against both primal and dual bounds. This audit concerns the transformation paths above; it does
not claim every dormant function or causal assumption in the repository has
been independently proved correct.

## Verification

After expanded zero support verification, the full source suite passed 331 tests
(20.11 seconds), including 26 support regression cases. Those cases include 450
seeded polynomial models checked at 25 exact rational points each, population
monotonicity permitting defiers, sensitivity parameter binding, nonlinear-to-linear
reduction, and primal/dual agreement. These finite checks supplement the targeted
regressions; they are not a proof for every polynomial model.

The expanded tests found and fixed a false successful DGP report for infeasible
models with no variables: the worker now checks SCIP's solution count before
extracting a witness. Models with and without remaining variables are covered.

Before this follow-up fix, both distribution formats built successfully and the
then-current 17 support cases passed against the installed wheel outside the checkout.
These checks used the existing Linux/Python 3.13 environment and its dependencies;
they do not establish a cross-platform compatibility matrix.

## Shared exact reductions and symbolic solving

`program.linearize()` (or `model.linearize()`) returns an immutable reduction
result without changing the source declarations:

```python
reduced = program.linearize(limits=(-1, 1))
print(reduced.report)
result = reduced.solve('scip', verbose=False, return_dgps=True)
# If jointly affine in decisions and symbolic parameters:
exact_bounds = reduced.solve('cdd')
```

The opt-in numerical shortcut is `program.run_scip(linearize=True)`.
`LinearResponseModel.solve(linearize=True)` and
`SymbolicProblem.solve(linearize=True)` use the same reductions before cdd.

The pipeline repeats exact zero elimination, affine equality substitution,
fixed-variable/fixed-sum reduction, and cancellation of factors proved
nonzero. Inequality cancellation additionally requires a proved sign. Bounds
on eliminated variables become constraints; reported numerical witnesses are
reconstructed in the original coordinates. A bounded-cost acceptance rule
limits expression growth and can conservatively skip useful substitutions.
Exact inconsistent equality systems are reported and left to the numerical
solver, avoiding amplification of tiny rounding residuals in input data.

For affine ratios, Charnes–Cooper is available when the remaining constraints
are linear and the denominator has certified strict sign and finite bounds.
The positive scale is explicitly bounded away from zero. Models with a
possibly zero denominator or symbolic parameter-dependent scaled coefficients
are left unchanged. These are sufficient reductions, not a guarantee that
all equivalent LPs will be recognized.

The usual response-mixture cdd backend remains the fast path. General jointly
affine models use exact polyhedral projection supporting arbitrary domains,
inequalities, and objective constants. That fallback enumerates the full
polyhedron's generators and can be expensive. Products of symbolic data and
decision variables are not silently treated as affine.


### Automatic parametric response reduction

`SymbolicProblem(...)` automatically selects reduction for observed independent
roots and event-based homogeneity declarations. It automatically constructs
a rational response-feature hull, substitutes equalities, and proves linearity
in decisions with data fixed. `write_program().linearize()` exposes the
feature map, hull constraints, reduced mathematical model, and report;
`solve()` eliminates nuisance coordinates with certified symbolic signs.
This path rejects unresolved nonlinearity or unknown elimination signs and
retains parameter compatibility conditions and zero-mass cases.


### Failures remain visible

Known symbolic capability limits raise `autobounds.UnsupportedModelError`
(a `ValueError` subclass for compatibility). The message identifies the
unsupported graph structure, unresolved product, coefficient-sign issue, or
other limitation. This does not mean the causal model is infeasible.

Malformed declarations, incompatible data, missing dependencies, and resource
limits retain their respective error messages. A projection-budget error
reports the configured limit and the size of the next elimination step.
Unexpected exceptions are not relabeled as unsupported models or silently
retried with another solver; their original traceback is preserved.

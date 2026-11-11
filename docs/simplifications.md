# Supported simplifications and transformation audit

`Program.simplify_linear()` and its string/SymPy replacement helpers have been
removed. They were not covered by active solver tests, recognized only a narrow
subset of linear equalities, and could discard defining equations after
substitution. A linear-looking expression is not sufficient evidence that an
elimination preserves the feasible set.

## Using the replacement

```python
program = problem.write_program()
report = program.simplify()
print(report)
result = program.run_scip(verbose=False)
```

The method changes the program in place and returns before/after constraint and
term counts and the names of fixed parameters. It is opt-in; `write_program()`
and `run_scip()` do not automatically run this pass. They retain their existing
constant-tautology cleanup.

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
has its own presolve. Retaining variables preserves their names for returned
DGPs and keeps their original bounds available to the solver.

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

The full source suite passed 292 tests (21.49 seconds); the final installed
wheel passed the same 292 tests outside the checkout (21.63 seconds).
The numerical subsampling example and exact symbolic factorial example also
completed. These checks used the existing Linux/Python 3.13 environment and
its dependencies; they do not establish a cross-platform compatibility matrix.

# Changelog

## Unreleased

- Add symbolic equality and inequality declarations through `add_assumption`;
  automatically eliminate proved zero types while retaining population
  constraints. Keep `add_homogeneity` as a separate declaration.

- Distinguish unsupported symbolic models with `UnsupportedModelError`; add
  regressions for different model structures, capability limits, invalid
  declarations, missing dependencies, and projection budgets.

- Select the symbolic solver automatically from model structure. Preserve the
  fast linear path, expose the selection reason, and propagate backend failures
  without silent fallback or partial results.

- Add opt-in exact reductions shared by numerical and symbolic solvers, with
  affine substitutions, sign-safe cancellation, Charnes–Cooper, and witness
  reconstruction. Add a general jointly affine cdd projection fallback.
- Add automatic symbolic response-feature reduction from graph, data, and
  event-based homogeneity declarations. Derive parametric LPs and eliminate
  nuisance coordinates with certified signs, including zero-mass strata.
  Validate against independently declared numerical response models.

- Fix c-component detection for overlapping latent parents and automatically
  enforce graph-implied response-block independences inside each component.
- Use a reduced independence basis and shared auxiliary marginals. Substitute
  marginals fixed by exact data equalities to avoid unnecessary nonlinear solving.
- Add regressions comparing overlapping-latent IV, ordinary IV, Robins–Manski,
  and common-confounder bounds for `P(Y(D=1)=1)`, including primal/dual checks.

## 0.1.0rc1 — release candidate (unpublished)

This candidate collects the architecture, correctness, performance, and
distribution changes since 0.0.2. See the [migration guide](docs/migration-0.1.md)
before upgrading existing scripts.

- Separate immutable mathematical models, solver backends, results, and
  inference workflows. Add exact linear symbolic bounds through pycddlib.
- Automatically eliminate provably zero decision variables during compilation
  and reconstruct their zero values in successful generated distributions.
- Replace unsafe simplification with conservative polynomial rewrites. Preserve
  constant contradictions and variable domains.
- Remove fixed solver polling delays, index parser lookups, cache symbolic
  bounds, and reuse subsampling setup.
- Use spawned SCIP workers across platforms, with lightweight package imports
  and cleanup on worker failure. Preserve primal and dual bound distinctions.
- Retire Pyomo/Couenne integrations and archive obsolete workflows.
- Split plotting and statistical inference dependencies into optional extras.
  Keep numerical solving and empirical subsampling in the base installation.
- Test installed wheels built from source distributions on Linux, macOS, and
  Windows with Python 3.10 and 3.13; test feature extras separately on Linux.
- Use one canonical version for package metadata and runtime version fields.

General nonlinear quantifier elimination and a Redlog backend are not included.

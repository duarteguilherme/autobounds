# Changelog

## Unreleased

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

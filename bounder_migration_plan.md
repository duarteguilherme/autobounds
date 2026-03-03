# Bounder Migration Plan

## Goal

Split responsibilities so:

- `Bounder` handles a single bounds problem lifecycle.
- `causalProblem` orchestrates one or many `Bounder` instances for higher-level workflows (CI, sensitivity, subgroup aggregation).

## PR1 Scope (Start Here)

1. Introduce a new `Bounder` class as a backward-compatible single-problem API.
2. Keep `causalProblem` behavior unchanged for current users.
3. Export `Bounder` from package top-level (`autobounds.__init__`).
4. Do not break current tests or user scripts.

## PR1 Implementation

1. Add `autobounds/bounder.py`:
   - `Bounder` subclasses `causalProblem`.
   - No behavior changes yet.
   - Add class docstring clarifying future role.

2. Update `autobounds/__init__.py`:
   - Export `Bounder`.
   - Include it in `__all__`.

3. Keep `causalProblem` as-is for now (compatibility-first).

## PR2 (Next)

1. Move single-problem internals from `causalProblem` into `Bounder` implementation (state + core methods).
2. Refactor `causalProblem` into orchestrator with:
   - implicit default bounder for backwards compatibility,
   - optional multiple bounders for advanced workflows.

## PR3 (Stabilization)

1. Add API wrappers/deprecations for old direct calls on orchestrator.
2. Update tests:
   - single-problem behavior in `test_bounder.py`,
   - orchestration in `test_causalproblem.py`.
3. Update docs/examples.

## PR3 Status

- Added explicit backward-compatible wrapper methods on `causalProblem` for common single-bounder APIs (`load_data`, `set_ate`, `solve`, etc.).
- Kept fallback proxying for uncovered attributes, with a deprecation warning to guide migration toward explicit `Bounder` usage.
- Updated README with `Bounder` vs `causalProblem` API guidance.

## Acceptance Criteria for PR1

1. `from autobounds import Bounder` works.
2. Existing `causalProblem` usage remains unchanged.
3. Test suite passes unchanged.

## Status

- PR1: completed.
- PR2: started with orchestration scaffolding in `causalProblem`:
  - bounder registry (`add_bounder`, `new_bounder`, `get_bounder`, `list_bounders`, `bounders`),
  - batch solving helper (`solve_bounders`),
  - compatibility preserved (default bounder is still the current object).

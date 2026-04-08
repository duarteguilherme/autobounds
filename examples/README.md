# Examples

This folder keeps the main coverage and CI comparison scripts:

## 1) Coverage with covariates (`covariate_ate_ci_coverage.py`)

Use this for coverage checks when `read_data(..., covariates=[...])` is used.

```bash
python covariate_ate_ci_coverage.py --r 20 --n 1200 --b 500 --workers 16 --ci-workers 16 --subsample-rate 0.6666667 --print-cis
```

Notes:
- uses `causalProblem` syntax (`set_ate`, `read_data`),
- CI uses subsampling via `causalProblem.solve(ci=True)` (same subsampling controls as no-covariate path).
- `--maxtime` is the per-optimization SCIP time limit in seconds.
- reports coverage for aggregate bounds and per-covariate-level (`X=x`) bounds.

## 2) Coverage without covariates (`no_covariate_ate_ci_coverage.py`)

Use this for no-covariate coverage checks via subsampling CI in `causalProblem.solve`.

```bash
python no_covariate_ate_ci_coverage.py --r 75 --n 600 --b 500 --ci-workers 16 --subsample-rate 0.6666667 --print-cis
```

Notes:
- `--maxtime` is the per-optimization SCIP time limit in seconds.

## 3) Closed-form Balke-Pearl lower-bound CI comparison (`balke_pearl_closed_form_ci_compare.py`)

Use this when you want a pure closed-form Monte Carlo comparison of:
- an oracle normal CI that knows the true active lower-bound formula,
- the current recentered subsampling rule with `gamma = 2/3` and minimum subsample size `80`,
- a standard percentile bootstrap.

```bash
python balke_pearl_closed_form_ci_compare.py --model oracle_friendly --r 500 --b 500 --workers 16
python balke_pearl_closed_form_ci_compare.py --model nearly_tied --r 500 --b 500 --workers 16
```

Notes:
- loops over `n = 100, 200, 300, 500, 1000, 2000` by default,
- uses exact population Balke-Pearl lower candidates from a binary latent IV SCM, so no solver is involved,
- `oracle_friendly` has a clearly separated active lower-bound formula,
- `nearly_tied` has a very small gap between the best two lower-bound formulas, which is useful for studying poor separability,
- default `--b` is `500`,
- writes a summary CSV plus coverage/width/active-match/point-estimate PNG plots to `examples/outputs/`.

# Examples

This folder keeps only the two coverage scripts:

## 1) Coverage with covariates (`covariate_ate_ci_coverage.py`)

Use this for coverage checks when `read_data(..., covariates=[...])` is used.

```bash
python covariate_ate_ci_coverage.py --r 75 --n 600 --nsamples 100 --print-cis
```

Notes:
- uses `causalProblem` syntax (`set_ate`, `read_data`),
- CI is delegated to the default `Bounder` because `causalProblem.solve(ci=True)` with covariates is still deferred.
- `--maxtime` is the per-optimization SCIP time limit in seconds.

## 2) Coverage without covariates (`no_covariate_ate_ci_coverage.py`)

Use this for no-covariate coverage checks via subsampling CI in `causalProblem.solve`.

```bash
python no_covariate_ate_ci_coverage.py --r 75 --n 600 --b 500 --ci-workers 16 --print-cis
```

Notes:
- `--maxtime` is the per-optimization SCIP time limit in seconds.

# Examples

This folder has IV + ATE scripts with and without covariates.

## 1) Single-run CI (`iv_ate_ci_subsampling.py`)

Use this when you want bounds + CI for one simulated dataset.

```bash
python iv_ate_ci_subsampling.py --n 1500 --nsamples 80 --ci-workers 16 --subsample-rate 0.7 --ci-method recentered_subsampling --maxtime 15
```

What it does:
- simulates one IV dataset,
- sets ATE,
- computes point bounds,
- computes subsampling CIs.

## 2) Coverage study (`iv_ate_ci_coverage.py`)

Use this when you want empirical CI coverage over many simulated datasets.

```bash
python iv_ate_ci_coverage.py --r 1000 --b 500 --n 600 --workers 8 --ci-workers 16 --subsample-rate 0.7 --ci-method recentered_subsampling --maxtime 4
```

What it does:
- approximates true bounds from a large simulated dataset,
- repeats `R` times: simulate data, compute CI using `B` subsamples,
- reports lower/upper/joint coverage.

## 3) Coverage with covariates (`covariate_ate_ci_coverage.py`)

Use this when you want coverage in a setting with covariates and `read_data(..., covariates=[...])`.

```bash
python covariate_ate_ci_coverage.py --r 75 --n 600 --nsamples 100 --print-cis
```

Notes:
- this script uses `causalProblem` syntax for setup (`set_ate`, `read_data`),
- CI computation is delegated to the default `Bounder` since covariate CI in `causalProblem.solve(ci=True)` is currently deferred.

## 4) Coverage without covariates (`no_covariate_ate_ci_coverage.py`)

Use this when you want the same compact coverage loop style as the covariate script, but with no covariates.

```bash
python no_covariate_ate_ci_coverage.py --r 75 --n 600 --b 500 --ci-workers 16 --print-cis
```

## Which one should I use?

- Use `iv_ate_ci_subsampling.py` for one-off analysis.
- Use `iv_ate_ci_coverage.py` for simulation/validation of CI performance.
- Use `covariate_ate_ci_coverage.py` for covariate coverage checks.
- Use `no_covariate_ate_ci_coverage.py` for the same style coverage loop without covariates.

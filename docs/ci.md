# Clean-install CI

The `Installed wheel tests` workflow runs on pushes to `main`, pull requests,
and manual dispatch. Numerical jobs cover Linux, macOS, and Windows with
Python 3.10 and 3.13. Four Linux/Python 3.13 jobs install `plot`, `inference`,
`symbolic`, and `plot,inference,symbolic`, respectively. Jobs including symbolic
support install cddlib/GMP development libraries. There are ten jobs in total.

Every job builds an sdist and a wheel from it, creates a fresh virtual
environment without system packages, installs that wheel and its dependencies,
and runs `pip check`. Tests and Python example scripts are copied to a temporary
directory outside the checkout. The harness verifies the imported package lives
inside the fresh environment before running the suite. `PYTHONPATH` and
`PYTHONHOME` are removed, and user-site packages are disabled. Resolved dependency
versions are printed in the job log.

The suite includes actual spawned SCIP solves, both primal and dual bounds,
returned distributions, zero-stratum reconstruction, worker construction errors,
timeouts, cleanup, import behavior, and causal/inference regressions. Numerical
jobs explicitly verify that plotting, statistical, cddlib, and tqdm packages
are absent. Feature jobs explicitly import the requested dependencies before
testing so a missing feature cannot silently produce a successful all-skipped
run. Tests needing unavailable features skip; numerical solving and empirical
subsampling still run in the minimal environment. Plotnine's transitive
statistical dependencies are allowed in the plotting job. Independent SciPy
comparisons of symbolic bounds run in the combined-extras job; the symbolic-only
job still checks exact bounds against SCIP through the shared-model test.

To reproduce locally, install `build` and run:

```sh
python ci/test_wheel.py
python ci/test_wheel.py --extras plot
python ci/test_wheel.py --extras inference
python ci/test_wheel.py --extras symbolic
python ci/test_wheel.py --extras plot,inference,symbolic
```

The symbolic invocations require cddlib/GMP development libraries. These commands
need network access for fresh build/runtime dependency resolution. Temporary
environments and copied tests are cleaned up after success or failure. These
jobs test their listed Python/platform combinations; they do not establish
support for every Python version or hardware architecture.

Before the optional-dependency split, the [first successful full matrix](https://github.com/duarteguilherme/autobounds/actions/runs/42954997770)
tested commit `12a7f33`: all seven jobs passed. The symbolic job passed 343 tests;
numerical installations passed 321 tests and skipped 22 requiring cddlib. This
run exposed and led to fixes for temporary-path aliases and the case-colliding
`Bounder.py`/`bounder.py` filenames. The lowercase compatibility import now lives
in a package directory, preserving both import styles on all tested systems.

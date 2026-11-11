# Clean-install CI

The `Installed wheel tests` workflow runs on pushes to `main`, pull requests,
and manual dispatch. Numerical jobs cover Linux, macOS, and Windows with
Python 3.10 and 3.13. A separate Linux/Python 3.13 job installs the symbolic
extra and cddlib/GMP development libraries.

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
jobs explicitly verify cddlib is absent; tests needing that optional backend
skip. The symbolic job explicitly imports `cdd.gmp` before testing so a missing
backend cannot silently produce a successful all-skipped run.

To reproduce locally, install `build` and run:

```sh
python ci/test_wheel.py
python ci/test_wheel.py --symbolic
```

The symbolic invocation requires cddlib/GMP development libraries. Both commands
need network access for fresh build/runtime dependency resolution. Temporary
environments and copied tests are cleaned up after success or failure. These
jobs test their listed Python/platform combinations; they do not establish
support for every Python version or hardware architecture.

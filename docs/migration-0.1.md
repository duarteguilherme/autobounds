# Migrating to 0.1

## Installation

Python 3.10 or newer is required. CI exercises Python 3.10 and 3.13 on Linux,
macOS, and Windows. Other Python versions are not covered by that matrix.

The base package provides numerical SCIP solving and empirical subsampling.
Request `autobounds[plot]` for plotting, `autobounds[inference]` for Gaussian/KL
and regression-based inference, or `autobounds[symbolic]` for exact linear
symbolic bounds. Combine extras as `autobounds[plot,inference,symbolic]`.
The symbolic extra may require native cddlib/GMP development libraries to build.
Existing environments retain previously installed dependencies; test a fresh
environment when checking whether your application declares all its extras.

## Script entry points

SCIP solving now starts workers using multiprocessing's spawn mechanism.
Put script execution behind a main guard so workers can import the module
without recursively starting more solves:

```python
def main():
    # Construct your problem and invoke its solver here.
    ...


if __name__ == "__main__":
    main()
```

Native SCIP model objects stay inside their worker processes. Legacy inspection
models do not represent the solved worker state; use returned solver results.

## Retired APIs and simplification

Pyomo and Couenne methods now raise `NotImplementedError`. Use `run_scip()`
for numerical bounds and `to_pip()` for polynomial program export. CIP export
is unsupported. The old `simplify_linear` method has been removed;
`Program.simplify()` performs conservative polynomial simplification.

Compilation automatically removes variables that constraints prove are zero.
This includes individual-level restrictions expressed as zero probabilities
when the polynomial representation establishes that each affected term is
nonnegative. An ambiguous product such as `x*y = 0` does not by itself justify
removing either variable. Population contrasts are not automatically treated
as individual-level exclusions.

Use `Program.compile(eliminate_zeros=False)` to inspect the unreduced model.
Successful generated distributions reconstruct eliminated variables as zero.
Explicit user exclusions and other mutating operations are separate from this
compilation pass.

## Imports and versions

Public package imports remain available and load on demand. Both historical
`autobounds.Bounder` and `autobounds.bounder` import paths remain supported.
`autobounds.__version__` now includes prerelease suffixes and matches installed
distribution metadata. `version_info` and the human-readable `version` remain
available; release candidates use `releaselevel="candidate"`.

The symbolic backend handles supported linear response models. It is not a
general nonlinear quantifier-elimination solver.

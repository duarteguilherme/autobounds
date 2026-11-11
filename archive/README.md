# Historical material

These files preserve old workflows and provenance; they are not supported
examples or installation instructions for the current package.

- `legacy_demos/`: former `autobounds_demo/` notebooks and their data.
  Seven of ten notebooks failed before the first solver call during the
  November 2026 audit (removed `autobounds.Query` or `Bounder.query` APIs).
  Two reached a solver and one was an import/graph template; this does not
  establish that their complete workflows still run. Notebook relative paths
  may require the original directory layout. Outputs have not been regenerated.
- `docker/`: original Ubuntu 20.04/Python 3.8 recipe and SCIP source archive.
  Python 3.8 is below the current package minimum of 3.10. The recipe also
  expects an externally supplied repository ZIP. It was not rebuilt.
- `legacy_tests/`: old analysis script requiring the removed `autobound` API,
  missing local data, and a hard-coded author output path. It is not a test.

Published replication scripts, data, and the frozen implementation remain in
`../replication_files/`. They require their historical environment; the frozen
code is not the installed package. Only generated caches, editor backups and
solver logs were removed there. Do not treat it as a source of current APIs.

Use `../examples/` and the root README for current interfaces.

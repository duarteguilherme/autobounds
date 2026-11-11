# Preparing a release

The current candidate is **0.1.0rc1**. Preparation does not upload distributions
or create a release tag.

1. Update the literal `__version__` in `autobounds/_version.py`. Setuptools reads
   this same value for wheel and source-distribution metadata.
2. Review `CHANGELOG.md` and the migration guide. For a final release, remove
   the candidate suffix and record the publication date only when publishing.
3. Run the complete suite with all feature dependencies installed:

   ```bash
   python -m pip install -e '.[dev,plot,inference,symbolic]'
   python -m pytest
   ```

4. Build into a new, version-specific directory and check metadata:

   ```bash
   python -m pip install build twine
   python -m build --outdir dist/0.1.0rc1
   python -m twine check --strict dist/0.1.0rc1/*.whl dist/0.1.0rc1/*.tar.gz
   python ci/test_wheel.py
   python ci/test_wheel.py --extras plot,inference,symbolic
   ```

   The harness builds a source distribution, builds its wheel, then installs
   and tests that wheel outside the checkout. It verifies that runtime and
   installed metadata versions match. Each invocation uses a fresh environment.

5. Require the installed-wheel CI jobs to pass for the exact release commit.
   The matrix covers Python 3.10/3.13 on three operating systems; this does not
   establish compatibility with every dependency version allowed by metadata.
6. Review the archive contents and retain checksums of the exact files intended
   for upload. Do not mix old distributions into the upload directory.

After approval to publish, upload those reviewed files to TestPyPI, verify an
installation of the exact candidate, then publish the intended version to PyPI
and create its matching Git tag/release. Do not change files after validation
without rebuilding and repeating the checks. Publication credentials and a
release tag are not configured by this preparation procedure.

The build requires setuptools 77.0.3 or newer because the metadata uses an SPDX
license expression; see the [setuptools license migration guide](https://setuptools.pypa.io/en/stable/userguide/license_migration.html).

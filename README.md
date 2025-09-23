# Autobounds

Autobounds is an advanced software tool designed to automate the calculation of partial identification bounds in causal inference problems (we're still in beta, so expect many changes soon).

Developed by researchers at the University of Pennsylvania, Johns Hopkins University, and Princeton University, Autobounds leverages polynomial programming and dual relaxation techniques to compute the sharpest possible bounds on causal effects, even when data is incomplete or mismeasured. You can learn more in their research paper.

This tool is particularly valuable in fields like economics, political science, and epidemiology, where researchers often face challenges such as confounding, selection bias, measurement error, and noncompliance. By automating the process, Autobounds allows for more precise and reliable estimation of causal relationships, facilitating better-informed decision-making.

To get started, download the software via Docker and run the following command to launch Autobounds:
docker run -p 8888:8888 -it gjardimduarte/autolab:v5
This will allow you to easily integrate Autobounds into your causal inference workflows.

To install it in your machine, clone this repo and use python -m pip install .

Development is currently being conducted by Guilherme Duarte, Dean Knox, and Kai Cooper. Code contributions were also made by Lisa Schulze-Bergkamen and Jeremy Zucker.

## Install from source

Use a recent Python (>=3.8) and a clean virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

This installs Autobounds in editable mode for development. For a regular install, replace the last line with:

```bash
python -m pip install .
```

## Build and publish (PyPI)

Build source and wheel distributions:

```bash
python -m pip install build twine
python -m build
```

Optional: upload to TestPyPI to verify packaging (requires an API token):

```bash
python -m twine upload -r testpypi dist/*
```

Publish to PyPI (requires a PyPI token):

```bash
python -m twine upload dist/*
```

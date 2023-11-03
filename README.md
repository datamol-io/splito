<div align="center">
    <img src="docs/images/logo-black.svg" height="150px">
    <h3>Splito - Dataset splitting for life sciences</h3>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/splito)](https://pypi.org/project/splito/)
[![Conda](https://img.shields.io/conda/v/conda-forge/splito?label=conda&color=success)](https://anaconda.org/conda-forge/splito)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/splito)](https://pypi.org/project/splito/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/splito)](https://anaconda.org/conda-forge/splito)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/splito)](https://pypi.org/project/splito/)
[![Code license](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/datamol-io/splito/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/splito)](https://github.com/datamol-io/splito/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/splito)](https://github.com/datamol-io/splito/network/members)

[![test](https://github.com/datamol-io/splito/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/test.yml)
[![release](https://github.com/datamol-io/splito/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/splito/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/splito/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/doc.yml)

Splito is a machine learning dataset splitting library for life sciences.

## Installation

You can install `splito` using pip:

```bash
pip install splito
```

You can use conda/mamba. Ask @maclandrol for credentials to the conda forge or for a token

```bash
mamba install -c conda-forge splito
```

## Documentation

Find the documentation at <https://splito-docs.datamol.io/>.

## Development lifecycle

### Setup dev environment

```bash
micromamba create -n splito -f env.yml
micromamba activate splito

pip install --no-deps -e .
```

### Tests

You can run tests locally with:

```bash
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

# Partitio

[![test](https://github.com/datamol-io/partitio/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/partitio/actions/workflows/test.yml)
[![release](https://github.com/datamol-io/partitio/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/partitio/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/partitio/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/partitio/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/partitio/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/partitio/actions/workflows/doc.yml)

Partitio is a machine learning dataset splitting library for life sciences.

## Development lifecycle

### Setup dev environment

```bash
micromamba create -n partitio -f env.yml
micromamba activate partitio

pip install -e .
```

### Tests

You can run tests locally with:

```bash
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

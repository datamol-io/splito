<div align="center">
    <img src="docs/images/logo-black.svg" height="150px">
    <h3>Splito - Dataset splitting for life sciences</h3>
</div>

---

[![test](https://github.com/datamol-io/splito/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/test.yml)
[![release](https://github.com/datamol-io/splito/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/splito/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/splito/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/splito/actions/workflows/doc.yml)

Splito is a machine learning dataset splitting library for life sciences.

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

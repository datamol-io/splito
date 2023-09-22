# Overview

Splito is a python library designed for aiding in drug discovery by providing powerful methods for parsing and splitting datasets. It enables researchers and chemists to efficiently process data for their ML projects.

Splito is part of the Datamol ecosystem: <https://datamol.io>.

## Installation

Use conda:

```bash
mamba install -c conda-forge splito
```

_**Note:** We highly recommend using a [Conda Python distribution](https://github.com/conda-forge/miniforge) to install Datamol. The package is also pip installable if you need it: `pip install splito`._

## Quick API Tour

```python
import datamol as dm
from splito import ScaffoldSplit


# Load some data
data = dm.data.chembl_drugs()

# Initialize a splitter
splitter = ScaffoldSplit(smiles=data["smiles"].tolist(), n_jobs=-1, test_size=0.2, random_state=111)

# Generate indices for training set and test set
train_idx, test_idx = next(splitter.split(X=data.smiles.values))
```

## Tutorials

Check out the [tutorials](tutorials/The_Basics.ipynb) to get started.

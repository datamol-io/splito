# Overview

Splito is a python library designed for aiding in drug discovery by providing powerful methods for parsing and splitting datasets. It enables researchers and chemists to efficiently process data for their ML projects.

Visit our website at <https://datamol.io>.

## Installation

Use conda:

```bash
mamba install -c conda-forge datamol
```

_**Tips:** You can replace `mamba` by `conda`._

_**Note:** We highly recommend using a [Conda Python distribution](https://github.com/conda-forge/miniforge) to install Datamol. The package is also pip installable if you need it: `pip install datamol`._

## Quick API Tour

```python
import datamol as dm

# Common functions
data = dm.data.chembl_drugs() 

# Define scaffold split
splitter = ScaffoldSplit(smiles=data.smiles.tolist(), n_jobs=-1, test_size=0.2, random_state=111)

# Generate index for training set and test set
train_idx, test_idx = next(splitter.split(X=data.smiles.values))

```

## How to cite

Please cite Datamol if you use it in your research: [![DOI](TO BE ADDED)]().

## Compatibilities

Version compatibilities are an essential topic for production-software stacks. We are cautious about documenting compatibility between `splito`, `datamol`, `python` and `pymoo`.

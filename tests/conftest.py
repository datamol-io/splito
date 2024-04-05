import datamol as dm
import numpy as np
import pytest


@pytest.fixture(scope="module")
def test_dataset():
    data = dm.data.freesolv()
    data["mol"] = [dm.to_mol(smi) for smi in data["smiles"]]
    data = data.dropna()
    return data


@pytest.fixture(scope="module")
def test_dataset_smiles(test_dataset):
    return test_dataset["smiles"].values


@pytest.fixture(scope="module")
def test_dataset_targets(test_dataset):
    return test_dataset["expt"].values


@pytest.fixture(scope="module")
def test_dataset_features(test_dataset):
    return np.array([dm.to_fp(mol) for mol in test_dataset["mol"].values])


@pytest.fixture(scope="module")
def test_deployment_set():
    data = dm.data.solubility()
    data["mol"] = [dm.to_mol(smi) for smi in data["smiles"]]
    data = data.dropna()
    return data


@pytest.fixture(scope="module")
def test_deployment_smiles(test_deployment_set):
    return test_deployment_set["smiles"].values


@pytest.fixture(scope="module")
def test_deployment_features(test_deployment_set):
    return np.array([dm.to_fp(mol) for mol in test_deployment_set["mol"].values])


@pytest.fixture(scope="module")
def manual_smiles():
    return [
        "CCCCC",
        "C1=CC=CC=C1",
        "CCCCOC(C1=CC=CC=C1)OCCCC",
        "CC1=CC(=CC(=C1O)C)C(=O)C",
        "CCN(CC)S(=O)(=O)C1=CC=C(C=C1)C(=O)OCC",
        "C[Si](C)(C)CC1=CC=CC=C1",
        "CN1C=NC2=C1C(=O)NC(=O)N2C",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]

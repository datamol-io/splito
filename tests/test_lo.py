import numpy as np
from rdkit.Chem import AllChem
import datamol as dm
from rdkit import Chem, DataStructs

from splito.lohi import LoSplitter


def test_lo(test_dataset_smiles, test_dataset_targets):
    # deafult parameters
    splitter = LoSplitter()
    train_idx, clusters_idx = splitter.split(test_dataset_smiles, test_dataset_targets)

    for cluster in clusters_idx:
        one_cluster_check(train_idx, cluster, test_dataset_smiles, 0.4, 5, test_dataset_targets, 0.60)

    # different parameters
    splitter = LoSplitter(threshold=0.6, min_cluster_size=7, std_threshold=0.8)
    train_idx, clusters_idx = splitter.split(test_dataset_smiles, test_dataset_targets)

    for cluster in clusters_idx:
        one_cluster_check(train_idx, cluster, test_dataset_smiles, 0.6, 7, test_dataset_targets, 0.80)


def one_cluster_check(train_idx, cluster_idx, smiles, threshold, min_cluster_size, values, std_threshold):
    assert len(cluster_idx) >= min_cluster_size

    # Ensure there is only one similar molecule in the train
    train_smiles = smiles[train_idx]
    cluster_smiles = smiles[cluster_idx]
    distance_matrix = dm.similarity.cdist(cluster_smiles, train_smiles, radius=2, nBits=1024)
    similarity_matrix = 1.0 - distance_matrix
    is_too_similar = similarity_matrix > threshold
    no_hits_per_mol = np.sum(is_too_similar, axis=1)
    assert np.array_equal(no_hits_per_mol, np.ones(len(cluster_smiles), dtype=int))

    # Assert the hit is the same for all cluster molecules.
    hit_indices = np.argmax(is_too_similar, axis=1)
    assert (hit_indices == hit_indices[0]).all()

    # Verify the variation within the cluster exceeds the threshold
    hit_smiles = train_smiles[hit_indices[0]]
    hit_idx = list(smiles).index(hit_smiles)
    cluster_values = np.append(values[cluster_idx], values[hit_idx])
    assert cluster_values.std() >= std_threshold

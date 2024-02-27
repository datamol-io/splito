import pytest
import numpy as np
from rdkit.Chem import AllChem
import datamol as dm
from rdkit import Chem, DataStructs

from splito.lohi import LoSplitter

def test_lo():
    data = dm.solubility()
    data = data.drop_duplicates(['smiles'])

    smiles = []
    val = []

    for idx in range(len(data)):
        if AllChem.MolFromSmiles(data.iloc[idx]['smiles']):
            smiles.append(data.iloc[idx]['smiles'])
            val.append(data.iloc[idx]['SOL'])

    smiles = np.array(smiles)
    val = np.array(val)

    # deafult parameters
    splitter = LoSplitter()
    train_idx, clusters_idx = splitter.split(smiles, val)

    for cluster in clusters_idx:
        one_cluster_check(train_idx, cluster, smiles, 0.4, 5, val, 0.60)
    

    # different parameters
    splitter = LoSplitter(threshold=0.6, min_cluster_size=7, std_threshold=0.8)
    train_idx, clusters_idx = splitter.split(smiles, val)

    for cluster in clusters_idx:
        one_cluster_check(train_idx, cluster, smiles, 0.6, 7, val, 0.80)


def one_cluster_check(train_idx, cluster_idx, smiles, threshold, min_cluster_size, values, std_threshold):
    assert len(cluster_idx) >= min_cluster_size

    # Ensure there is only one similar molecule in the train
    train_smiles = smiles[train_idx]
    train_mol = [Chem.MolFromSmiles(s) for s in train_smiles]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_mol]

    cluster_smiles = smiles[cluster_idx]
    cluster_mol = [Chem.MolFromSmiles(s) for s in cluster_smiles]
    cluster_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in cluster_mol]

    hit_indices = set()
    for cluster_f in cluster_fps:
        sims = DataStructs.BulkTanimotoSimilarity(cluster_f, train_fps)
        sims = np.array(sims)
        assert sum(sims > threshold) == 1
        hit_indices.add(np.argmax(sims))
    
    assert len(hit_indices) == 1  # the hit is the same for all cluster
    
    hit_idx = None
    for idx in hit_indices:
        hit_idx = idx
    hit_smiles = train_smiles[hit_idx]
    hit_idx = list(smiles).index(hit_smiles)

    cluster_values = list(values[cluster_idx]) + [values[hit_idx]]
    cluster_values = np.array(cluster_values)
    
    assert cluster_values.std() >= std_threshold



from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm

from ._utils import get_similar_mols

class LoSplitter():
    def __init__(self, threshold=0.4, min_cluster_size=5, max_clusters=50, std_threshold=0.60):
        """
        Parameters:
            threshold -- molecules with a similarity higher than this value are considered similar.
            min_cluster_size -- number of molecules per cluster.
            max_clusters -- maximum number of selected clusters. The remaining molecules go to the training set.
            std_threshold -- Lower bound of the acceptable standard deviation for a cluster. It should be greater than measurement noise.
                            If you're using ChEMBL-like data, set it to 0.60 for logKi and 0.70 for logIC50.
                            Set it lower if you have a high-quality dataset.
        """
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.std_threshold = std_threshold

    def split(self, smiles, values, verbose=1):
        """
        Split the dataset of smiles and their continuous values.

        Parameters:
            smiles -- list of smiles.
            values -- list of their continuous activity values.
            verbose -- set to 0 to turn off progressbar.

        Returns:
            train_idx -- list of train indices.
            clusters_idx -- list of lists of cluster indices.
        """
        if not isinstance(smiles, np.ndarray):
            smiles = np.array(smiles)
        if smiles.shape != np.unique(smiles).shape:
            raise ValueError("Remove duplicates from your smiles")

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        train_idx, clusters_idx = self._select_distinct_clusters(
            smiles, values, verbose
        )
        train_idx = list(train_idx)

        # Move one molecule from each cluster to the training set
        leave_one_clusters = []
        for cluster in clusters_idx:
            train_idx.append(cluster[-1])
            leave_one_clusters.append(cluster[:-1])
        
        if not leave_one_clusters:
            print('No clusters were found. Was your std_threshold too constrained?')

        return train_idx, leave_one_clusters

    def _select_distinct_clusters(
        self, smiles, values, verbose
    ):
        """
        A greedy algorithm to select independent clusters from datasets.
        """
        if verbose:
            progress_bar = tqdm(total=self.max_clusters, desc="Collecting clusters")

        train_smiles = smiles.copy()
        all_clusters = []

        mols = [Chem.MolFromSmiles(smile) for smile in train_smiles]
        all_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
        while len(all_clusters) < self.max_clusters:
            total_neighbours, stds = self._get_neighborhood(all_fps, values)
            central_idx = self._get_central_idx(total_neighbours, stds)
            if central_idx is None:
                break  # there are no more clusters
           
            cluster_idx = self._collect_cluster(central_idx, all_fps)
            cluster_smiles = train_smiles[cluster_idx]
            all_clusters.append(cluster_smiles)

            # Remove neighbours of the cluster from the rest of smiles
            nearest_sim = get_similar_mols(train_smiles, cluster_smiles)
            rest_fps = []
            rest_idx = []
            for idx, sim in enumerate(nearest_sim):
                if sim < self.threshold:
                    rest_idx.append(idx)
                    rest_fps.append(all_fps[idx])
            all_fps = rest_fps
            train_smiles = train_smiles[rest_idx]
            values = values[rest_idx]

            if verbose:
                progress_bar.update(1)
        
        if verbose:
            progress_bar.close()
            print(f'Found {len(all_clusters)} clusters.')
        
        smile_to_idx = dict()
        for idx, smile in enumerate(smiles):
            smile_to_idx[smile] = idx
        
        train_smiles_idx = [smile_to_idx[s] for s in train_smiles]
        all_clusters_idx = []
        for cluster in all_clusters:
            cluster_idx = [smile_to_idx[s] for s in cluster]
            all_clusters_idx.append(cluster_idx)

        return train_smiles_idx, all_clusters_idx
    
    def _get_neighborhood(self, all_fps, values):
        """
        For each molecule find number of neighbours and std of their values.
        """
        total_neighbours = []
        stds = []
        for fps in all_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fps, all_fps)
            is_neighbor = np.array(sims) > self.threshold
            total_neighbours.append(is_neighbor.sum())
            stds.append(values[is_neighbor].std())

        total_neighbours = np.array(total_neighbours)
        stds = np.array(stds)
        return total_neighbours, stds
    
    def _get_central_idx(self, total_neighbours, stds):
        """
        Find the most distant cluster and return the centroid of it.
        """
        central_idx = None
        least_neighbours = max(total_neighbours)
        for idx, n_neighbours in enumerate(total_neighbours):
            if n_neighbours > self.min_cluster_size:
                if n_neighbours < least_neighbours:
                    if stds[idx] >= self.std_threshold:
                        least_neighbours = n_neighbours
                        central_idx = idx
        return central_idx
    
    def _collect_cluster(self, central_idx, all_fps):
        """
        Collect list of neighbours of the central_idx.
        """
        sims = DataStructs.BulkTanimotoSimilarity(all_fps[central_idx], all_fps)
        is_neighbour = np.array(sims) > self.threshold
        cluster_idx = []
        for idx, value in enumerate(is_neighbour):
            if value:
                if (
                    idx != central_idx
                ):  # we add the central molecule at the end of the list
                    cluster_idx.append(idx)
        cluster_idx.append(central_idx)
        return cluster_idx

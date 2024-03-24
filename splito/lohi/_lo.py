from rdkit import DataStructs
import numpy as np
from tqdm import tqdm
import datamol as dm
import functools
from loguru import logger


class LoSplitter:
    def __init__(self, threshold=0.4, min_cluster_size=5, max_clusters=50, std_threshold=0.60):
        """
        Creates the splitter object.

        Args:
            threshold: molecules with a similarity higher than this value are considered similar.
            min_cluster_size: number of molecules per cluster.
            max_clusters: maximum number of selected clusters. The remaining molecules go to the training set.
            std_threshold: Lower bound of the acceptable standard deviation for a cluster. It should be greater than measurement noise.
                            If you're using ChEMBL-like data, set it to 0.60 for logKi and 0.70 for logIC50.
                            Set it lower if you have a high-quality dataset.
        """
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.std_threshold = std_threshold

    def split(self, smiles, values, n_jobs=-1, verbose=1):
        """
        Split the dataset of smiles and their continuous values.

        Args:
            smiles: list of smiles.
            values: list of their continuous activity values.
            verbose: set to 0 to turn off progressbar.

        Returns:
            train_idx: list of train indices.
            clusters_idx: list of lists of cluster indices.
        """
        if not isinstance(smiles, np.ndarray):
            smiles = np.array(smiles)
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        train_idx, clusters_idx, central_nodes = self._select_distinct_clusters(
            smiles, values, n_jobs, verbose
        )
        train_idx = list(train_idx) + central_nodes

        if not clusters_idx:
            logger.warninig("No clusters were found. Was your std_threshold too constrained?")

        return train_idx, clusters_idx

    def _select_distinct_clusters(self, smiles, values, n_jobs, verbose):
        """
        A greedy algorithm to select clusters from neighborhood graph of molecules.
        """
        if verbose:
            progress_bar = tqdm(total=self.max_clusters, desc="Collecting clusters")

        # At first, all the nodes are in the train set. Some will be moved to the list of clusters.
        train_nodes = np.array(range(len(smiles)))

        train_fps = dm.parallelized(
            functools.partial(dm.to_fp, as_array=False, radius=2, nBits=1024),
            smiles,
            n_jobs=n_jobs,
        )
        all_clusters_nodes = []  # the test clusters of nodes
        central_nodes = []  # central nodes of the clusters

        while len(all_clusters_nodes) < self.max_clusters:
            total_neighbours, stds = self._get_neighborhood(train_fps, values)
            central_idx = self._get_central_idx(total_neighbours, stds)
            if central_idx is None:
                break  # there are no more clusters
            central_nodes.append(train_nodes[central_idx])

            cluster_indices = self._collect_cluster(central_idx, train_fps)

            # Save the cluster nodes
            all_clusters_nodes.append(train_nodes[cluster_indices])

            # Remove neighbours of the cluster from the rest of nodes
            nearest_sim = self._get_nearest_sim(train_fps, cluster_indices + [central_idx])
            rest_idx = []
            for idx, sim in enumerate(nearest_sim):
                if sim < self.threshold:
                    rest_idx.append(idx)
            train_nodes = train_nodes[rest_idx]
            values = values[rest_idx]
            train_fps = [train_fps[idx] for idx in rest_idx]

            if verbose:
                progress_bar.update(1)
        if verbose:
            progress_bar.close()
        logger.info(f"Found {len(all_clusters_nodes)} clusters.")
        return train_nodes, all_clusters_nodes, central_nodes

    def _get_neighborhood(self, train_fps, values):
        """
        For each node find number of neighbours and std of their values.
        """
        total_neighbours = []
        stds = []
        for fps in train_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fps, train_fps)
            is_neighbor = np.array(sims) > self.threshold
            total_neighbours.append(is_neighbor.sum())
            stds.append(values[is_neighbor].std())

        total_neighbours = np.array(total_neighbours)
        stds = np.array(stds)
        return total_neighbours, stds

    def _get_central_idx(self, total_neighbours, stds):
        """
        Find the most distant cluster and return the index of its centroid.
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

    def _collect_cluster(self, central_idx, train_fps):
        """
        Collect list of neighbours of the central_idx.
        """
        sims = DataStructs.BulkTanimotoSimilarity(train_fps[central_idx], train_fps)
        is_neighbour = np.array(sims) > self.threshold
        cluster_indices = []
        for idx, value in enumerate(is_neighbour):
            if value:
                if idx != central_idx:
                    cluster_indices.append(idx)
        return cluster_indices

    def _get_nearest_sim(self, train_fps, indices_to_remove):
        """
        For each train molecule find the maximal similarity to molecules in the cluster_smiles.
        """
        cluster_fps = [train_fps[idx] for idx in indices_to_remove]
        nearest_sim = []
        for train_fp in train_fps:
            sims = DataStructs.BulkTanimotoSimilarity(train_fp, cluster_fps)
            nearest_sim.append(max(sims))
        return nearest_sim

import abc
from typing import Callable, Optional, Sequence, Union

import datamol as dm
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split  # noqa W0212
from sklearn.utils.validation import _num_samples  # noqa W0212

from .utils import get_kmeans_clusters

# In case users provide a list of SMILES instead of features, we rely on ECFP4 and the tanimoto distance by default
MOLECULE_DEFAULT_FEATURIZER = dict(name="ecfp", kwargs=dict(radius=2, fpSize=2048))
MOLECULE_DEFAULT_DISTANCE_METRIC = "jaccard"


def guess_distance_metric(example):
    """Guess the appropriate distance metric given an exemplary datapoint"""

    # By default we use the Euclidean distance
    metric = "euclidean"

    # For binary vectors we use jaccard
    if isinstance(example, pd.DataFrame):
        example = example.values  # DataFrames would require all().all() otherwise
    if ((example == 0) | (example == 1)).all():
        metric = "jaccard"

    return metric


def convert_to_default_feats_if_smiles(
    X: Union[Sequence[str], np.ndarray], metric: str, n_jobs: Optional[int] = None
):
    """
    If the input is a sequence of strings, assumes this is a list of SMILES and converts it
    to a default set of ECFP4 features with the default Tanimoto distance metric.
    """

    def _to_feats(smi: str):
        mol = dm.to_mol(smi)
        feats = dm.to_fp(
            mol=mol, fp_type=MOLECULE_DEFAULT_FEATURIZER["name"], **MOLECULE_DEFAULT_FEATURIZER["kwargs"]
        )
        return feats

    if all(isinstance(x, str) for x in X):
        X = dm.utils.parallelized(_to_feats, X, n_jobs=n_jobs)
        metric = MOLECULE_DEFAULT_DISTANCE_METRIC
    return X, metric


class DistanceSplitBase(GroupShuffleSplit, abc.ABC):
    """Base class for any splitter that splits the data based on the distance matrix."""

    def __init__(
        self,
        n_splits=10,
        metric: Optional[Union[str, Callable]] = None,
        n_jobs: Optional[int] = None,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._metric = metric
        self._n_jobs = n_jobs

    @abc.abstractmethod
    def get_split_from_distance_matrix(
        self, mat: np.ndarray, group_indices: np.ndarray, n_train: int, n_test: int
    ):
        """Abstract method that needs to be implemented by subclasses"""
        raise NotImplementedError

    def reduce(self, X: np.ndarray, split_idx: int):
        """
        Gives an endpoint for reducing the number of groups to
        make computing the distance matrix faster.
        """
        return X

    def compute_distance_matrix(self, groups: np.ndarray):
        """Allows subclasses to override the way the distance matrix is computed"""
        distance_matrix = pairwise_distances(groups, metric=self._metric, n_jobs=self._n_jobs)
        return distance_matrix

    def _iter_indices(
        self,
        X: Union[Sequence[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[Union[int, np.ndarray]] = None,
    ):
        """
        Generate (train, test) indices

        Specifically, it computes the distance matrix for the (possibly reduced groups of) samples.
        It then yields the train and test indices based on the distance matrix.

        If X is a list of SMILES, rather than features, the SMILES are converted to ECFP4 features and
        the Tanimoto distance metric is used by default.
        """

        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        base_seed = self.random_state
        if base_seed is None:
            base_seed = 0

        # Convert to ECFP4 if X is a list of smiles
        X, self._metric = convert_to_default_feats_if_smiles(X, self._metric, n_jobs=self._n_jobs)
        if self._metric is None:
            self._metric = guess_distance_metric(X[0])

        for i in range(self.n_splits):
            # Possibly group the data to improve computation efficiency
            groups = self.reduce(X, base_seed + i)

            # Compute the distance matrix
            unique_groups, group_indices, group_counts = np.unique(
                groups, return_inverse=True, return_counts=True, axis=0
            )
            distance_matrix = self.compute_distance_matrix(unique_groups)

            # Compute the split
            train, test = self.get_split_from_distance_matrix(
                mat=distance_matrix, group_indices=group_indices, n_train=n_train, n_test=n_test
            )
            yield train, test


class KMeansReducedDistanceSplitBase(DistanceSplitBase, abc.ABC):
    """
    Base class for any distance based split that reduced the samples using k-means clustering
    """

    def __init__(
        self,
        metric: Union[str, Callable] = "euclidean",
        n_clusters: int = 25,
        n_jobs: Optional[int] = None,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            n_jobs=n_jobs,
            metric=metric,
        )
        self._n_clusters = n_clusters

    def reduce(self, X: np.ndarray, split_idx: int):
        """
        Uses k-means to group the data and reduce the number of unique data points.
        In case the specified metric is not euclidean, we will use the Empirical Kernel Map to transform the features
        to a space that is euclidean compatible.
        """

        seed = None if self.random_state is None else self.random_state + split_idx

        _, groups = get_kmeans_clusters(
            X=X,
            n_clusters=self._n_clusters,
            random_state=seed,
            return_centers=True,
            base_metric=self._metric,
        )

        return groups

    def compute_distance_matrix(self, groups: np.ndarray):
        """Override the metric to always be euclidean due to the use of k-means"""
        original = self._metric
        self._metric = "euclidean"
        mat = super().compute_distance_matrix(groups)
        self._metric = original
        return mat

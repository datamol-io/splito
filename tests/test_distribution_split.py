import pytest

from splito import StratifiedDistributionSplit
from splito._distribution_split import Clustering1D


@pytest.mark.parametrize("algorithm", list(Clustering1D))
def test_splits_stratified_distribution(test_dataset_smiles, test_dataset_targets, algorithm):
    splitter = StratifiedDistributionSplit(algorithm=algorithm, n_splits=2)

    for train_ind, test_ind in splitter.split(test_dataset_smiles, y=test_dataset_targets):
        assert len(train_ind) + len(test_ind) == len(test_dataset_targets)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0

        # TODO (cwognum): Add more specialized tests

from sklearn.model_selection import ShuffleSplit

from splito import (
    MaxDissimilaritySplit,
    MolecularMinMaxSplit,
    MolecularWeightSplit,
    MOODSplitter,
    PerimeterSplit,
    ScaffoldSplit,
    StratifiedDistributionSplit,
)


def test_mood_split(
    test_dataset_smiles, test_dataset_features, test_dataset_targets, test_deployment_features
):
    splitters = {
        "random": ShuffleSplit(n_splits=1),
        "scaffold": ScaffoldSplit(test_dataset_smiles, n_splits=1),
        "perimeter": PerimeterSplit(metric="jaccard", n_clusters=5, n_splits=1),
        "max_dissimilarity": MaxDissimilaritySplit(metric="jaccard", n_clusters=5, n_splits=1),
        "min_max": MolecularMinMaxSplit(smiles=test_dataset_smiles, n_splits=1),
        "molecular_weight": MolecularWeightSplit(smiles=test_dataset_smiles, n_splits=1),
        "stratified_distribution": StratifiedDistributionSplit(n_clusters=5, n_splits=1),
    }

    splitter = MOODSplitter(splitters, metric="jaccard")
    splitter.fit(X=test_dataset_features, X_deployment=test_deployment_features, y=test_dataset_targets)

    for train_ind, test_ind in splitter.split(test_dataset_features, y=test_dataset_targets):
        train = test_dataset_features[train_ind]
        test = test_dataset_features[test_ind]

        assert len(train) > 0 and len(test) > 0
        assert len(train) + len(test) == len(test_dataset_features)

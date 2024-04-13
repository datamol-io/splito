import numpy as np
import pytest

from splito import train_test_split
from splito._split import SimpleSplittingMethod


@pytest.mark.parametrize("method", list(SimpleSplittingMethod))
@pytest.mark.parametrize("as_string", [True, False])
def test_train_test_split(
    method, as_string, test_dataset_smiles, test_dataset_targets, test_dataset_features
):
    if as_string:
        method = method.name.lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X=test_dataset_features, y=test_dataset_targets, method=method, molecules=test_dataset_smiles
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) + len(X_test) == len(test_dataset_features)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

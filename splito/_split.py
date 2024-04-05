from enum import Enum, unique
from typing import Optional, Sequence, Union

import datamol as dm
import numpy as np
from sklearn.model_selection import ShuffleSplit

from ._distribution_split import StratifiedDistributionSplit
from ._kmeans_split import KMeansSplit
from ._max_dissimilarity_split import MaxDissimilaritySplit
from ._min_max_split import MolecularMinMaxSplit
from ._molecular_weight import MolecularWeightSplit
from ._perimeter_split import PerimeterSplit
from ._scaffold_split import ScaffoldSplit


@unique
class SimpleSplittingMethod(Enum):
    RANDOM = ShuffleSplit
    KMEANS = KMeansSplit
    PERIMETER = PerimeterSplit
    MAX_DISSIMILARITY = MaxDissimilaritySplit
    SCAFFOLD = ScaffoldSplit
    STRATIFIED_DISTRIBUTION = StratifiedDistributionSplit
    MOLECULAR_WEIGHT = MolecularWeightSplit
    MIN_MAX_DIVERSITY_SPLIT = MolecularMinMaxSplit


def train_test_split_indices(
    X: np.ndarray,
    y: np.ndarray,
    molecules: Optional[Sequence[Union[str, dm.Mol]]] = None,
    method: Union[str, SimpleSplittingMethod] = "random",
    test_size: float = 0.2,
    seed: int = None,
    n_jobs: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the indices of the train and test sets.

    Different from scikit-learn's API, we assume some data-types are not represented as numpy arrays
    and cannot be directly indexed as we do in [`train_test_split`][splito.train_test_split]. This
    functions offers a way to just return the indices and take care of the split manually.

    See [`train_test_split`][splito.train_test_split] for more information.
    """
    X = np.array(X)
    y = np.array(y)

    method = SimpleSplittingMethod[method.upper()] if isinstance(method, str) else method

    splitter_kwargs = {"test_size": test_size, "random_state": seed}
    if method in [
        SimpleSplittingMethod.MOLECULAR_WEIGHT,
        SimpleSplittingMethod.MIN_MAX_DIVERSITY_SPLIT,
        SimpleSplittingMethod.SCAFFOLD,
    ]:
        if molecules is None:
            raise ValueError(f"{method.name} requires a list of molecules to be provided.")
        if isinstance(molecules[0], dm.Mol):
            molecules = dm.utils.parallelized(dm.to_smiles, molecules, n_jobs=n_jobs)
        splitter_kwargs["smiles"] = molecules

    splitter_cls = method.value
    splitter = splitter_cls(**splitter_kwargs)

    return next(splitter.split(X, y))


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    molecules: Optional[Sequence[Union[str, dm.Mol]]] = None,
    method: Union[str, SimpleSplittingMethod] = "random",
    test_size: float = 0.2,
    seed: int = None,
    n_jobs: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a set of molecules into a train and test set.

    Inspired by sklearn.model_selection.train_test_split, this function is meant as a convenience function
    that provides a less verbose way of using the different splitters.

    **Examples**:

     Let's first create a toy dataset

     ```python
     import datamol as dm
     import numpy as np

     data = dm.data.freesolv()
     smiles = data["smiles"].values
     X = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles])
     y = data["expt"].values
     ```

     Now we can split our data.

     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, method="random")
     ```

     More parameters
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, method="random", test_size=0.1, random_state=42)
     ```

     Scaffold split (note that you need to specify `smiles`):
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, smiles=smiles, method="scaffold")
     ```

     Distance-based split:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, method="kmeans")
     ```

    Args:
        X: The feature matrix.
        y: The target values.
        molecules: A list of molecules to be used for the split. Required for some splitting methods.
        method: The splitting method to use. Defaults to "random".
        test_size: The proportion of the dataset to include in the test split.
        seed: The seed to use for the random number generator.
        n_jobs: The number of jobs to run in parallel.
    """
    train_indices, test_indices = train_test_split_indices(
        X, y, molecules=molecules, method=method, test_size=test_size, seed=seed, n_jobs=n_jobs
    )
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

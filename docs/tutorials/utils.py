from typing import List

from matplotlib import pyplot as plt

import pandas as pd
import umap
import seaborn as sns
import datamol as dm
import warnings

warnings.filterwarnings("ignore")


def visualize_chemspace(data: pd.DataFrame, split_names: List[str], mol_col: str = "smiles", size_col=None):
    figs = plt.figure(num=3)
    features = [dm.to_fp(mol) for mol in data[mol_col]]
    embedding = umap.UMAP().fit_transform(features)
    data["UMAP_0"], data["UMAP_1"] = embedding[:, 0], embedding[:, 1]
    for split_name in split_names:
        plt.figure()
        fig = sns.scatterplot(data=data, x="UMAP_0", y="UMAP_1", style=size_col, hue=split_name, alpha=0.7)
        fig.set_title(f"UMAP Embedding of compounds for {split_name}")
    return figs

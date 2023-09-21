from matplotlib import pyplot as plt
import pandas as pd
import umap
import seaborn as sns
import datamol as dm
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings. filterwarnings('ignore')

def visualize_chemspace(data:pd.DataFrame, split_names:str, mol_col:str="smiles", size_col=None):
    figs = plt.figure(num=3)
    features = [dm.to_fp(mol) for mol in data[mol_col]]
    embedding = umap.UMAP().fit_transform(features)
    data["UMAP_0"], data["UMAP_1"] = embedding[:, 0], embedding[:, 1]
    for split_name in split_names:
        plt.figure()
        # if size_col is None:
        #     size = 10 
        # else:
        #     scaler = MinMaxScaler((10 , 50))
        #     size = scaler.fit_transform(data[[size_col]].values).flatten()

        fig = sns.scatterplot(
            data=data,
            x="UMAP_0",
            y="UMAP_1",
            style=size_col,
            hue=split_name,
            alpha=0.7
        )
        fig.set_title(f"UMAP Embedding of compounds for {split_name}")
    return figs
import numpy as np
import pandas as pd

#from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn import manifold

if __name__ == '__main__':

    import argparse
    import os

    script_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser(description="Use -f PATH/TO/DATASET to specifiy the dataset which should be transformed into a graph.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", help="path to dataset", type=str)
    parser.add_argument("-i", "--indexcol", help="specify whether index column is given or not (Y/n)", type=str)

    args = parser.parse_args()
    if args.indexcol == "n":
        df = pd.read_csv(args.file)
    else:
        df = pd.read_csv(args.file, index_col=0)

    methods = ["Isomap", "LocallyLinearEmbedding", "SpectralEmbedding", "TSNE"]
    names = ["isomap", "lle", "spectrale", "tsne"]

    dataset_name = args.file.split("/")[-1].split(".")[0]

    if len(df) > 2000:
        df = df.sample(2000)

    conts = []
    for col in df.columns:
        if str(df[col].dtype) == "float64" or str(df[col].dtype) == "int64":
            conts.append(col)
    print("Get continuous dimensions: ", conts, "\nStart embedding methods ...")

    for method, name in zip(methods, names):
        print("Start ", method, " algorithm ... ", end="")
        df_org = df
        df_work = pd.DataFrame(df_org, columns = conts)
        funct = getattr(manifold, method)
        embedding = funct(n_components=2)
        df_trans = embedding.fit_transform(df_work)
        df_org["Emb_dim1"] = df_trans[:,0]
        df_org["Emb_dim2"] = df_trans[:,1]
        string = "embedded_datasets/" + dataset_name + "_" + name + ".csv"
        df_org.to_csv(string)
        print("DONE")

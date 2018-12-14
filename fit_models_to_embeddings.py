import numpy as np
import pandas as pd

#from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn import manifold
from sklearn.model_selection import train_test_split
import mb_modelbase as mbase




if __name__ == '__main__':

    import argparse
    import os

    script_name = os.path.basename(__file__)

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Use -f PATH/TO/DATASET to specifiy the dataset which should be transformed into a graph.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", help="path to dataset", type=str)
    parser.add_argument(
        "-i", "--indexcol", help="specify whether index column is given or not (Y/n)", type=str)
    parser.add_argument("-m", "--methods", help="specify which embedding algorithms should be used (available: isomap, lle, spectrale, tsne, mds)", nargs='+', default=["isomap", "lle", "spectrale", "tsne", "mds"])
    parser.add_argument("-n", "--num_samples", help="Select a number of samples (default: 1000)", type=int, default=1000)

    args = parser.parse_args()

    # Specify whether input dataset has index column or not
    if args.indexcol == "n":
        df = pd.read_csv(args.file)
    else:
        df = pd.read_csv(args.file, index_col=0)

    # Parse methods for embeddings and assign them to list
    meth_nam = {"isomap": "Isomap", "lle": "LocallyLinearEmbedding", "spectrale": "SpectralEmbedding", "tsne": "TSNE", "mds": "MDS"}
    methods, names = [],[]
    for meth in args.methods:
        names.append(meth)
        methods.append(meth_nam[meth])

    #meth_nam = {"isomap": "Isomap", "lle": "LocallyLinearEmbedding", "spectrale": "SpectralEmbedding", "tsne": "TSNE", "mds": "MDS"}
    #methods = ["Isomap", "LocallyLinearEmbedding", "SpectralEmbedding", "TSNE", "MDS"]
    #names = ["isomap", "lle", "spectrale", "tsne", "mds"]

    # Exclude name of dataset for filename
    dataset_name = args.file.split("/")[-1].split(".")[0]

    # Use only a determined number of samples of the dataset
    if len(df) > args.num_samples:
        df = df.sample(args.num_samples)
        df.reset_index(inplace=True, drop=True)

    # Split in test and train set
    train, test = train_test_split(df, test_size=0.1)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    # Get continuous dimensions
    conts = []
    for col in df.columns:
        if str(df[col].dtype) == "float64" or str(df[col].dtype) == "int64":
            conts.append(col)

    # Start embeddings
    print("Get continuous dimensions: ", conts,
          "\nStart embedding methods ...")

    for method, name in zip(methods, names):
        string = dataset_name + "_" + name + "_pred"

        print("Start ", method, " algorithm ... ")
        df_org = train
        df_test = test
        df_work = pd.DataFrame(df_org, columns=conts)
        funct = getattr(manifold, method)
        embedding = funct(n_components=2)
        df_trans = embedding.fit_transform(df_work)
        df_org = df_org.assign(Emb_dim1=df_trans[:,0])
        df_org = df_org.assign(Emb_dim2=df_trans[:,1])
        df_org.to_csv("embedded_datasets/" + string + ".csv")

        print("Fit model ... ")
        mymod = mbase.MixableCondGaussianModel(string)
        mymod.fit(df=df_org, bool_test_data=False)

        print("Start predictions ... ")
        emb1, emb2 = [], []
        for row in df_test.iterrows():
            mymod_cond = mymod.copy()
            for col in df_test.columns:
                mymod_cond = mymod_cond.copy().condition(mbase.Condition(col, "==", row[1][col]))
            argmax = mymod_cond.aggregate("maximum")
            emb1.append(argmax[-2])
            emb2.append(argmax[-1])
        df_test = df_test.assign(Emb_dim1=emb1)
        df_test = df_test.assign(Emb_dim2=emb2)

        mymod.test_data = df_test
        mymod.save(model=mymod, filename="fitted_emb_pred/" + string + ".mdl")
        print("Saved model ", string, "successfully")

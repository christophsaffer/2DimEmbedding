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

    parser = argparse.ArgumentParser(
        description="Use -f PATH/TO/DATASET to specifiy the dataset which should be transformed into a graph.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", help="path to dataset", type=str)
    parser.add_argument(
        "-i", "--indexcol", help="specify whether index column is given or not (Y/n)", type=str)

    args = parser.parse_args()
    if args.indexcol == "n":
        df = pd.read_csv(args.file)
    else:
        df = pd.read_csv(args.file, index_col=0)

    methods = ["Isomap", "LocallyLinearEmbedding", "SpectralEmbedding", "TSNE", "MDS"]
    names = ["isomap", "lle", "spectrale", "tsne", "mds"]

    dataset_name = args.file.split("/")[-1].split(".")[0]

    if len(df) > 1000:
        df = df.sample(1000)
        df.reset_index(inplace=True, drop=True)

    train, test = train_test_split(df, test_size=0.1)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    conts = []

    for col in df.columns:
        if str(df[col].dtype) == "float64" or str(df[col].dtype) == "int64":
            conts.append(col)
    print("Get continuous dimensions: ", conts,
          "\nStart embedding methods ...")

    for method, name in zip(methods, names):
        print("Start ", method, " algorithm ... ", end="")
        df_org = train
        df_test = test
        df_work = pd.DataFrame(df_org, columns=conts)
        funct = getattr(manifold, method)
        embedding = funct(n_components=2)
        df_trans = embedding.fit_transform(df_work)
        df_org = df_org.assign(Emb_dim1=df_trans[:,0])
        df_org = df_org.assign(Emb_dim2=df_trans[:,1])
        print("DONE")
        print("Fit model ... ", end="")
        string = dataset_name + "_" + name + "_pred"
        #df_org.to_csv(string)
        mymod = mbase.MixableCondGaussianModel(string)
        mymod.fit(df=df_org, bool_test_data=False)
        print("DONE")
        print("Start predictions ... ", end="")
        emb1, emb2 = [], []
        for row in df_test.iterrows():
            mymod_cond = mymod.copy()
            for col in df_test.columns:
                mymod_cond = mymod_cond.copy().condition(mbase.Condition(col, "==", row[1][col]))
            argmax = mymod_cond.aggregate("maximum")
            emb1.append(argmax[-2])
            emb2.append(argmax[-1])
        print("DONE")
        df_test = df_test.assign(Emb_dim1=emb1)
        df_test = df_test.assign(Emb_dim2=emb2)

        mymod.test_data = df_test
        print(len(df_test), " --- ", len(df_org))
        mymod.save(model=mymod, filename="fitted_emb_pred/" + string + ".mdl")

        print("Saved model ", string, "successfully")

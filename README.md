# 2DimEmbedding
A project for 2-dimensional embeddings 

The package uses the scikit-learn algorithms for the embeddings and the lumen-org project (https://github.com/lumen-org) for fitting the models visualization

### Help Page

$ python fit_models_to_embeddings.py --help

+ -h --help:  help text
+ -f --file: path to dataset
+ -i --indexcol: index column
+ -m --methods: methods for embedding
+ -n --num_samples: upper number of samples


### EXAMPLE for Embeddings

Calculates for the iris dataset two new embedded dimensions, one time with tsne, second time with mds for a maximum number of 100 datapoints


$ python fit_models_to_embeddings.py -f datasets/iris.csv -i n -m tsne mds -n 100


The outputs are saved in embedded_datasets (datasets) and fitted_emb_pred (models).




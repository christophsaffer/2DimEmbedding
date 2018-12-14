# 2DimEmbedding
A project for 2-dimensional embeddings 

The package uses the scikit-learn algorithms for the embeddings and the lumen-org project (https://github.com/lumen-org) for fitting the models visualization

To create 2-dimensional embeddings with a fitted model:

=== Manual ===

<code>
$ python fit_models_to_embeddings.py --help
usage: fit_models_to_embeddings.py [-h] [-f FILE] [-i INDEXCOL]
                                   [-m METHODS [METHODS ...]] [-n NUM_SAMPLES]

Use -f PATH/TO/DATASET to specifiy the dataset which should be transformed into a graph.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  path to dataset
  -i INDEXCOL, --indexcol INDEXCOL
                        specify whether index column is given or not (Y/n)
  -m METHODS [METHODS ...], --methods METHODS [METHODS ...]
                        specify which embedding algorithms should be used
                        (available: isomap, lle, spectrale, tsne, mds)
  -n NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Select an upper number of samples (default: 1000)

</code>

=== EXAMPLE for Embeddings ===

Calculates for the iris dataset two new embedded dimensions, one time with tsne, second time with mds for a maximum number of 100 datapoints

<code>
$ python fit_models_to_embeddings.py -f datasets/iris.csv -i n -m tsne mds -n 100
</code>

The outputs are saved in embedded_datasets (datasets) and fitted_emb_pred (models).




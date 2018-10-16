# Content Based Recommender System

## What is this?
This is a song recommender system, which applies a content based recommendation algorithms to a dataset of one million interactions between playlists 57k playlists and 100k tracks. The system then recommends five songs each to a set of 10k playlists.

## Why?
The Recommender System was beeing developed for this [Kaggle Competition](https://www.kaggle.com/c/recommender-system-2017-challenge-polimi)

## Functionality

### Main

`Perfect Content Based.ipynb` is the main file. It contains:
* a `Translator ()` object for creating and accessing index/attribute key pairs for every song attribute such as artist, album, tags etc..
* a `Data ()` class used to easily load the input data, filter it and split into train- and test sets, but also do build/load fundamental data structures such as the Item-Content-Matrix (ICM) and Item-Similarity-Matrix (ISM).
* a `Recommender ()` class, which is the recommender model used to fit based on input data and make predictions (recommendations) for a set of input data.
* an `Evaluator ()` class, which holds the evaluation functions which returns a score based on some predictions.

### Support functions
The Item-Similarity-Matrix is to large to compute on the Jupyter Notebook Python Kernel on a Macbook Laptop with 16GB ram. In order to overcome this, there are some support functions. `Build.py` calls `top_k.py` which increases data sparsity by only keeping data on the top k (5000 < k < 10000 works fine) most similar items for every item. The items were handled sequentially and intermediary results saved to file (`.npz`-format), and only merged into one full matrix in the very end in the file `merge.py`. In this way, the matrix can be computed on a laptop. The full result is saved to disk and can be loaded from the notebook (takes about 90s). 


## Time

* Building the Item-Similarity Matrix takes about 1h, but only needs to be done once.
* Fitting the model takes about 2h
* Predictions for 10k playlists take about five minutes.


By Philip Claesson and Miguel Maricalva.

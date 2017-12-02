# Do all relevant imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sps
import time
#%matplotlib inline
print("Libs imported.")

def apply_shrinkage(X, dist, shrinkage):
    # create an "indicator" version of X (i.e. replace values in X with ones)
    X_ind = X.copy()
    X_ind.data = np.ones_like(X_ind.data)
    # compute the co-rated counts
    co_counts = X_ind * X_ind.T
    # remove the diagonal
    co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
    # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
    # then multiply dist with it
    co_counts_shrink = co_counts.copy()
    co_counts_shrink.data += shrinkage
    co_counts.data /= co_counts_shrink.data
    dist.data *= co_counts.data
    return dist

shrinkage = 20
starttime = time.time()
cp = time.time()

ICM = sps.load_npz("Saved Matrixes/ICM_perfect.npz")
print("Loaded ICM! %s sec." %(time.time()-cp))
cp = time.time()

ISM = sps.load_npz("Saved Matrixes/ISM_perfect.npz")
print("Loaded ISM! %s sec." %(time.time()-cp))
cp = time.time()

if shrinkage > 0:
    dist = apply_shrinkage(ICM, ISM, shrinkage)
    print("Applied shrinkage %s sec. "%(time.time()-cp))
    cp = time.time()


sps.save_npz("Saved Matrixes/ISM_perfect_shrink", ISM)
print("Saved ISM! Total Time: %s " %(time.time()-starttime))

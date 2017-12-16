import numpy as np
import scipy.sparse as sps
import time

k = 5000
print("Reading Matrix")
M = sps.load_npz("Saved Matrixes/ISM_top_k_%s_coo.npz"%k)

print("Converting to .csr")
M_csr = M.tocsr()
#M_top_k = M_top_k.tocsr()
print("Saving matrix...")
outfilename = "Saved Matrixes/ISM_top_k_%s_csr.npz"%(k)
sps.save_npz(outfilename, M_csr)
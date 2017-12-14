import time
import numpy as np
import scipy.sparse as sps

def merge_to_ISM(totsize, stepsize, totsteps, k):
    rows = np.zeros(totsize*k)
    cols = np.zeros(totsize*k)
    vals = np.zeros(totsize*k)
    cp = time.time()
    for step in range(totsteps):
        filename = 'output/arrays_%s_%s.npz'%(k, step*stepsize)
        npzfile = np.load(filename)
        rows[step*k*stepsize:(step+1)*k*stepsize] = npzfile['arr_0']
        cols[step*k*stepsize:(step+1)*k*stepsize] = npzfile['arr_1']
        vals[step*k*stepsize:(step+1)*k*stepsize] = npzfile['arr_2']
        if step % 1 == 0:
            print("File %s out of %s. %s secs. " %(step, (totsize/stepsize), (time.time()-cp)))
            cp = time.time()



    print(vals.shape)
    print(rows.shape)
    print(cols.shape)
    print("Should be k vals per row, which means 100.000 k")
    M_top_k = sps.coo_matrix((vals, (rows, cols)), shape = (totsize, totsize), dtype = float).tocsr()
    print("Saving matrix...")
    outfilename = "Saved Matrixes/ISM_top_k_%s.npz"%(k)
    sps.save_npz(outfilename, M_top_k)
    print("Matrix saved!")
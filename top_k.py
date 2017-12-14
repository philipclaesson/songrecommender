def top_k(k, start, size):
    import numpy as np
    import scipy.sparse as sps
    import time
    starttime = time.time()

    #mask = np.zeros(M.shape, dtype = bool)
    rows = np.zeros((k*size,), dtype = np.int32)
    cols = np.zeros((k*size,), dtype = np.int32)
    vals = np.zeros((k*size,), dtype = np.float16)
    top_k_idx = np.zeros((k), dtype = np.int8) #Make higher if k > 127
    top_k_vals = np.zeros((k), dtype = np.int8) #Make higher if k > 127

    i = 0
    cp = time.time()
    print("Reading matrix..")
    M = sps.load_npz("Saved Matrixes/ISM_perfect_float_1000.npz")
    print("Loaded Matrix, %s sec." %(time.time()-cp))
    if(isinstance(M, sps.csr_matrix)):
        print("M is CSR")
    else:
        print("M is not CSR.. ")
    cp = time.time()
    index = start
    while index < start+size:

        track_similarities = M[index, :].todense()

        # Get top K indices. Sorting the whole thing will be faster than taking the
        # biggest val k times.
        # k loop: O(kn) or sorting: O(n log(n)) with n = 100.000 --> logn = 5.
        # sorting will be faster for all k greater than 5.
        top_k_idx = np.argsort(track_similarities)[0, -k:]
        top_k_vals = track_similarities[0, top_k_idx]


        # Create filter
        rows[k*i:k*(i+1)] = [i]*k # i indicates row
        cols[k*i:k*(i+1)] = top_k_idx
        vals[k*i:k*(i+1)] = top_k_vals
        if index % 500 == 0:
            print("Computed row %s out of %s in %s"%(index, size, (time.time()-cp)))
            #print(top_k_idx)
            #print(top_k_vals)
            cp = time.time()
        index += 1
        i += 1


    filename = 'output/arrays_%s_%s.npz'%(k, start)
    np.savez(filename, rows, cols, vals)
    print("Saved row %s to %s in %s sec. "%(start, start+size-1, time.time()-starttime))

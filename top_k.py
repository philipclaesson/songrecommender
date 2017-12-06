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
    M = sps.load_npz("Saved Matrixes/ISM_perfect_float.npz")
    print("Loaded Matrix, %s sec." %(time.time()-cp))
    check this!! is it right? print(isinstance(M, sps.csc_matrix))
    cp = time.time()
    index = start
    while index < start+size:

        track_similarities = M[index, :].todense()

        # Get top K indices
        for j in range(k):
            top_k_idx[j] = np.argmax(track_similarities)
            top_k_vals[j] = track_similarities[0, top_k_idx[j]]
            track_similarities[0, top_k_idx[j]] = 0.0



        # Create filter
        rows[k*i:k*(i+1)] = [i]*k
        cols[k*i:k*(i+1)] = top_k_idx
        top_k_vals = track_similarities[0, top_k_idx]
        vals[k*i:k*(i+1)] = top_k_vals
        if index % 50 == 0:
            print("Computed row %s out of %s in %s"%(index, size, (time.time()-cp)))
            cp = time.time()
        index += 1
        i += 1


    filename = 'output/arrays_%s_%s.npz'%(k, start)
    np.savez(filename, rows, cols, vals)
    print("Saved row %s to %s in %s sec. "%(start, start+size, time.time()-starttime))

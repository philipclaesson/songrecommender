import numpy as np
import time
cimport numpy as np
from cpython.array cimport array, clone

cdef get_rcv(M, k):
    #Takes dense matrix, returns rows, cols, vals.

    # Initiate np arrays.
    cdef np.ndarray[np.int32_t, ndim=1] rows = np.zeros(M.shape[0]*k, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] cols = np.zeros(M.shape[1]*k, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] vals = np.zeros(M.shape[0]*k, dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] top_k_vals = np.zeros(k, dtype=np.float32)

    i = 0
    for track_similarities in M:
        # Get top K indices
        top_k_idx = np.argsort(track_similarities)[-k:]

        # Create new matrix
        for j in range(k):
            rows[k*i+j] = i


        for j in range(len(top_k_idx)):
            cols[k*i+j] = top_k_idx[j]


        for j in range(k):
            top_k_vals[j] = track_similarities[top_k_idx[j]]


        for j in range(len(top_k_vals)):
            vals[k*i+j] = top_k_vals[j]


        i += 1
        if (i % 100 == 0):
            print(i)

    return rows, cols, vals
cp = time.time()
a = np.random.rand(100000,100000)
get_rcv(a, 20)
print(time.time()-cp)
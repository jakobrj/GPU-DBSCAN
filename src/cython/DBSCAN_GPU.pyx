import numpy as np
cimport numpy as np
cimport cython

# np.import_array()

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

### GPU-DBSCAN

cdef extern from "../cpp_wrappers/DBSCAN_GPU.cpp":
    void GPU_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts)

cdef GPU_DBSCAN_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, float eps, int minPts
):
    GPU_DBSCAN_cpp(&C[0], &data[0,0], n, d, eps, minPts)

def GPU_DBSCAN(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    float eps, int minPts
):
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    GPU_DBSCAN_cython(C, data, n, d, eps, minPts)
    return np.asarray(C)


### G-DBSCAN

cdef extern from "../cpp_wrappers/DBSCAN_GPU.cpp":
    void G_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts)

cdef G_DBSCAN_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, float eps, int minPts
):
    G_DBSCAN_cpp(&C[0], &data[0,0], n, d, eps, minPts)

def G_DBSCAN(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    float eps, int minPts
):
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    G_DBSCAN_cython(C, data, n, d, eps, minPts)
    return np.asarray(C)
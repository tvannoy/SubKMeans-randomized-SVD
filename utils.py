import numpy as np
import fbpca
from time import perf_counter
import multiprocessing as mp
import ctypes

# calculate the pc projection matrix
def calc_pc(dim, m):
    top = np.eye(m)
    bot = np.zeros((dim-m, m))
    pc = np.vstack((top, bot))
    return pc

# calculate the pn projection matrix
def calc_pn(dim, m):
    top = np.zeros((m, dim-m))
    bot = np.eye(dim-m)
    pn = np.vstack((top, bot))
    return pn

# initialize the transformation matrix
def init_transform(dim, m=None):
    if m is not None:
        V = np.random.rand(dim, m)
    else:
        V = np.random.rand(dim, dim)
        # we want V to be orthonormal, so it is initialized with
        # a QR decomposition of a random matrix
        V, r = np.linalg.qr(V)

    return V

# calculate scatter matrix as the inner product of the centered data matrix with itself
def calculate_scatter(data, num_processes=mp.cpu_count()):

    # Compute the mean vector
    mean = np.mean(data, axis=0)

    # The following shared memory approach was adapted from:
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

    # created shared memory RawArrays for the data matrices
    shape = data.shape
    centered_data = mp.RawArray(ctypes.c_double, data.size)
    scatter = mp.RawArray(ctypes.c_double, shape[1]*shape[1])

    # wrap the centered data matrix shared memory in a numpy array
    centered_data_np = np.frombuffer(centered_data, dtype=np.float64).reshape(shape)

    # copy data to the shared memory array
    np.copyto(centered_data_np, data - mean)

    # split the inner product up among the processes in the pool
    columns = np.array_split(np.arange(shape[1]), num_processes)
    with mp.Pool(processes=num_processes, initializer=_init_calculate_scatter_worker, initargs=(centered_data, scatter, shape)) as pool:
        pool.map(_calculate_scatter_worker, columns)

    return np.frombuffer(scatter, dtype=np.float64).reshape(shape[1], shape[1])

# intialize shared state for process pool
init = {}
def _init_calculate_scatter_worker(centered_data, scatter, shape):
    init['centered_data'] = centered_data
    init['scatter'] = scatter
    init['shape'] = shape

# compute part of the scatter matrix
def _calculate_scatter_worker(columns):
    # wrap the shared memory in numpy arrays
    centered_data_np = np.frombuffer(init['centered_data'], dtype=np.float64).reshape(init['shape'])
    scatter_np = np.frombuffer(init['scatter'], dtype=np.float64).reshape(init['shape'][1], init['shape'][1])

    # compute the partial inner product of the full data matrix with the sub-matrix formed by columns
    scatter_np[:,columns] = centered_data_np.T @ centered_data_np[:,columns]


def sorted_eig(s, m=None):
    # eigendecomposition -> this is where we will sub in randomized svd
    if m is not None:
        e_vals, e_vecs = fbpca.eigens(s, k=m)
    else:
        e_vals, e_vecs = np.linalg.eigh(s)

    # sort in ascending order
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:,idx]
    e_vals = e_vals[idx]

    return e_vals, e_vecs

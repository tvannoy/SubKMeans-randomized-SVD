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
centered_data = np.array([])
def calculate_scatter(data, num_processes=mp.cpu_count()):

    # Compute the mean vector
    mean = np.mean(data, axis=0)

    # The following shared memory approach was adapted from:
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    # https://stackoverflow.com/a/37746961
    # https://stackoverflow.com/a/43680177

    shape = data.shape

    # created a shared memory RawArray for the scatter matrix
    scatter = mp.RawArray(ctypes.c_double, shape[1]*shape[1])

    global centered_data
    centered_data = data - mean

    # split the inner product up among the processes in the pool
    columns = np.array_split(np.arange(shape[1]), num_processes)
    with mp.Pool(processes=num_processes, initializer=_init_calculate_scatter_worker, initargs=(scatter, shape)) as pool:
        pool.map(_calculate_scatter_worker, columns)

    return np.frombuffer(scatter, dtype=np.float64).reshape(shape[1], shape[1])

# intialize shared state for process pool
init = {}
def _init_calculate_scatter_worker(scatter, shape):
    init['scatter'] = scatter
    init['shape'] = shape

# compute part of the scatter matrix
def _calculate_scatter_worker(columns):
    # wrap the shared memory in a numpy array
    # NOTE: we don't need a lock on this memory because each process writes to disjoint regions
    scatter_np = np.frombuffer(init['scatter'], dtype=np.float64).reshape(init['shape'][1], init['shape'][1])

    # compute start and end indices to slice into the arrays; we do this
    # because slices are much more efficient than using array indexing
    start = columns[0]
    end = columns[-1] + 1

    # compute the partial inner product of the full data matrix with the sub-matrix formed by columns
    scatter_np[:,start:end] = centered_data.T @ centered_data[:,start:end]


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

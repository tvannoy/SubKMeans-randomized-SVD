import numpy as np
import fbpca
from time import perf_counter
import multiprocessing as mp

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
    t0 = perf_counter()

    if m is not None:
        V = np.random.rand(dim, m)
    else:
        V = np.random.rand(dim, dim)
        # we want V to be orthonormal, so it is initialized with
        # a QR decomposition of a random matrix
        V, r = np.linalg.qr(V)

    t1 = perf_counter()
    print('\tinit_transform runtime: {}'.format(t1-t0))
    return V

# calculate scatter matrix as the inner product of the centered data matrix
def calculate_scatter(data, num_processes=mp.cpu_count()):
    t0 = perf_counter()
    # Compute the mean vector
    mean = np.mean(data, axis=0)

    # Compute centered data matrix
    centered_data = data - mean

    # split centered data matrix into num_processes sub-matrices of
    # approximately equal size
    subarrays = np.array_split(centered_data, num_processes, axis=1)

    # split the inner product up among the processes in the pool, then
    # concatenate the results back together
    with mp.Pool(num_processes) as p:
        S = np.concatenate(p.starmap(np.matmul, [(centered_data.T, B) for B in subarrays]), axis=1)

    t1 = perf_counter()
    print('\tcalculate_scatter runtime: {}'.format(t1-t0))

    return S

def sorted_eig(s, m=None):
    t0 = perf_counter()

    # eigendecomposition -> this is where we will sub in randomized svd
    if m is not None:
        e_vals, e_vecs = fbpca.eigens(s, k=m)
    else:
        e_vals, e_vecs = np.linalg.eigh(s)

    # sort in ascending order
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:,idx]
    e_vals = e_vals[idx]

    t1 = perf_counter()
    print('\tsorted_eig runtime: {}'.format(t1-t0))
    return e_vals, e_vecs

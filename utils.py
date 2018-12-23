import numpy as np
import fbpca
from time import perf_counter

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

# calculate the scatter matrix for the full dataset
def calculate_scatter(data):
    t0 = perf_counter()

    #Compute the mean vector
    mean = np.mean(data, axis=0)

    #Computation of scatter plot
    s_d = (data - mean).T @ (data - mean)

    t1 = perf_counter()
    print('\tcalculate_scatter runtime: {}'.format(t1-t0))
    # d = len(data)
    # if d == 0:
    #     return [0]
    # c = np.eye(d) - np.multiply(1/d, np.ones((d,d)))
    # s_d = data.T @ c
    # s_d = s_d @ data
    return s_d

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

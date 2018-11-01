import numpy as np
import fbpca
import pickle 
from numba import jit 

# calculate the pc projection matrix
def calc_pc(dim, m):
    top = np.eye(m)
    bot = np.zeros((dim-m, m))
    pc = np.vstack((top, bot))
    return top

# calculate the pn projection matrix
def calc_pn(dim, m):
    top = np.zeros((m, dim-m))
    bot = np.eye(dim-m)
    pn = np.vstack((top, bot))
    return top

# initialize the transformation matrix
def init_transform(m, dim):
    V = np.random.rand(dim, m)
    # we want V to be orthonormal, so it is initialized with
    # a QR decomposition of a random matrix
    #V, r = np.linalg.qr(V)
    return V

# calculate the scatter matrix for the full dataset
def calculate_scatter(data):
    d = len(data)
    if d == 0:
        return [0]
    c = np.eye(d) - np.multiply(1/d, np.ones((d,d)))
    s_d = data.T @ c
    s_d = s_d @ data
    return s_d

def sorted_eig(s, m, randomized=False):
    # eigendecomposition -> this is where we will sub in randomized svd
    if randomized:
        # k = len(s)
        e_vals, e_vecs = fbpca.eigens(s, k=m)
    else:
        e_vals, e_vecs = np.linalg.eig(s)

    # sort in ascending order
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:,idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs

def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
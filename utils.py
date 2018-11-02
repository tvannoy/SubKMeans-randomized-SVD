import numpy as np
import fbpca 

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

# calculate the scatter matrix for the full dataset
def calculate_scatter(data):
    d = len(data)
    if d == 0:
        return [0]
    c = np.eye(d) - np.multiply(1/d, np.ones((d,d)))
    s_d = data.T @ c
    s_d = s_d @ data
    return s_d

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

import numpy as np 

def calc_pc(dim, m):
    top = np.eye(m)
    bot = np.zeros((dim-m, m))
    pc = np.vstack((top, bot))
    return pc

def init_transform(dim):
    V = np.random.rand(dim, dim)
    # we want V to be orthonormal, so it is initialized with 
    # a QR decomposition of a random matrix
    V, r = np.linalg.qr(V)
    return V
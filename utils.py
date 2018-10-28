import numpy as np

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
def init_transform(dim):
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
    # s = np.zeros((data.shape[1], data.shape[1]))

    # for i in range(len(data)):
    #     s += (data[i, :] - mean) @ (data[i, :] - mean).T

    return s_d

def eigen_decomp(s_i, s_d):
    # eigendecomposition -> this is where we will sub in randomized svd
    w, v = np.linalg.eig(s_i - s_d)
    return w, v

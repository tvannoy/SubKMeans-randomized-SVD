import numpy as np 

# calculate the pc projection matrix
def calc_pc(dim, m):
    top = np.eye(m)
    bot = np.zeros((dim-m, m))
    pc = np.vstack((top, bot))
    return pc

# initialize the transformation matrix
def init_transform(dim):
    V = np.random.rand(dim, dim)
    # we want V to be orthonormal, so it is initialized with 
    # a QR decomposition of a random matrix
    V, r = np.linalg.qr(V)
    return V

# calculate the scatter matrix for the full dataset 
def calculate_scatter(data, mean):
    s = np.zeros((data.shape[1], data.shape[1]))

    for i in range(len(data)):
        s += (data[i, :] - mean) @ (data[i, :] - mean).T 

    return s 

def eigen_decomp(s_i, s_d):
    # eigendecomposition -> this is where we will sub in randomized svd
    w, v = np.linalg.eig(s_i - s_d)

    # sort eigenvalues in ascending order
    sorted_v = [x for _,x in sorted(zip(w,v))]

    # create new transformation matrix
    return (np.vstack(sorted_v).T, w, v)
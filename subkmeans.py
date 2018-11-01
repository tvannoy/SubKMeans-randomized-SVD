import numpy as np
import utils
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
import os 
import time 
from numba import jit 


class SubKmeans(object):
    def __init__(self, k, data):
        self.k = k                                           # number of clusters
        self.data = data
        self.m = int(np.sqrt(data.shape[1]))                 # cluster space dims
        self.transform = utils.init_transform(self.m, data.shape[1])        # init transformation matrix
        self.pc = []                                         # projection matrix
        # compute dataset mean -> used in scatter matrix
        self.data_mean = np.mean(data, axis=0)

        # computer scatter matrix S_D
        self.s_d = utils.calculate_scatter(self.data)

        # track cluster assignments with dict. cluster num is key
        self.assignments = defaultdict(list)

        # cluster assignments in the cluster subspace
        self.cluster_space_assignments = defaultdict(list)

        # cluster assignments in the noise subspace
        self.noise_space_assignments = defaultdict(list)

        # randomly assign points to clusters initially.
        # Initial cluster centroids are chosen using random data points
        init_centroid_idx = np.random.choice(len(data), k, replace=False)
        self.centroids = self.data[init_centroid_idx, :]

    def run(self, max_iter=1000, randomized=False, tol=0.99):
        t0 = time.time()
        self._find_cluster_assignment()
        t1 = time.time()
        print("find cluster assignment: {}".format(t1-t0))

        t0 = time.time()
        self._update_centroids()
        t1 = time.time()
        print("update centroids: {}".format(t1-t0))

        t0 = time.time()
        self._update_transformation()
        t1 = time.time()
        print("update transformation: {}".format(t1-t0))

        n = 0
        nmi = 0
        while (n < max_iter and nmi < tol):
            prev_assignments = self.assignments
            # get previous labels
            prev_labels = []
            for k, v in prev_assignments.items():
                prev_labels += list(k * np.ones(len(v)))

            self._find_cluster_assignment()
            self._update_centroids()
            self._update_transformation(randomized=randomized)

            cur_labels = []
            for k,v in self.assignments.items():
                cur_labels += list(k * np.ones(len(v)))

            nmi = normalized_mutual_info_score(prev_labels, cur_labels)

            if n % 50 == 0:
                print("Iteration: {}".format(n))
                print(nmi)

            n += 1
            if nmi > tol:
                print(nmi,n)

        # save the cluster assignments
        if not os.path.isdir("Cur_Run"):
            os.mkdir("Cur_Run")

        utils.pickle_it(self.assignments, "Cur_Run/assignments.p")
        utils.pickle_it(self.transform, "Cur_Run/transform.p")
        utils.pickle_it(self.pc, "Cur_Run/pc.p")
        print("Cluster space dims: {}".format(self.m))


    def _update_centroids(self):
        for k, v in self.assignments.items():
            pts = np.vstack(v)
            centroid = np.mean(pts, axis=0)
            self.centroids[k,:] = centroid

    def _find_cluster_assignment(self):
        # re initialize clusters, as we are creating new assignments
        self.assignments = defaultdict(list)
        self.cluster_space_assignments = defaultdict(list)
        self.noise_space_assignments = defaultdict(list)

        # calculate the cluster space mapping
        #self.pc = utils.calc_pc(self.data.shape[1], self.m)   
        #cluster_space_mapping = self.pc.T @ self.transform.T  

        # calculate the noise space mapping
        #self.pn = utils.calc_pn(self.data.shape[1], self.m)
        #noise_space_mapping = self.pn.T @ self.transform.T

        # map data to cluster space
        mapped_data = (self.transform.T @ self.data.T).T # (i by m) i being the datapoints

        # map centroids to cluster space
        mapped_centroids = (self.transform.T @ self.centroids.T).T  # (k by m) k being number of centroids

        # map data to noise space
        #noise_data = (noise_space_mapping @ self.data.T).T

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(mapped_centroids - mapped_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.assignments[cluster_assignment].append(self.data[i,:])
            self.cluster_space_assignments[cluster_assignment].append(mapped_data[i,:])
            #self.noise_space_assignments[cluster_assignment].append(noise_data[i,:])

    def _update_transformation(self, randomized=False):
        # compute scatter matrix
        s_i = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in range(self.k):
            s_i += (utils.calculate_scatter(np.array(self.assignments[i])))

        # where we sub in randomized svd
        eigen_values, self.transform = utils.sorted_eig(s_i - self.s_d, self.m, randomized=randomized)
        self._get_M(eigen_values)

    def _get_M(self, eigen_values):
        self.m = max(1, len([i for i in eigen_values if i < -1e-10]))

import numpy as np
import utils
from collections import defaultdict
# from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kmeans import Kmeans


class SubKmeans(Kmeans):
    def __init__(self, k, data):
        super().__init__(k, data)
        self.m = int(np.sqrt(data.shape[1]))                 # cluster space dims
        self.transform = utils.init_transform(data.shape[1]) # init transformation matrix
        self.pc = []                                         # projection matrix
        self.s_d = utils.calculate_scatter(self.data)        # compute scatter matrix S_D
        self.cluster_space_assignments = defaultdict(list)   # cluster assignments in the cluster subspace
        self.noise_space_assignments = defaultdict(list)     # cluster assignments in the noise subspace

    def _find_cluster_assignment(self):
        # re initialize clusters, as we are creating new assignments
        self.assignments = defaultdict(list)
        self.cluster_space_assignments = defaultdict(list)
        self.noise_space_assignments = defaultdict(list)

        # calculate the cluster space mapping
        self.pc = utils.calc_pc(self.data.shape[1], self.m)
        cluster_space_mapping = self.pc.T @ self.transform.T

        # calculate the noise space mapping
        self.pn = utils.calc_pn(self.data.shape[1], self.m)
        noise_space_mapping = self.pn.T @ self.transform.T

        # map data to cluster space
        mapped_data = (cluster_space_mapping @ self.data.T).T # (i by m) i being the datapoints

        # map centroids to cluster space
        mapped_centroids = (cluster_space_mapping @ self.centroids.T).T  # (k by m) k being number of centroids

        # map data to noise space
        noise_data = (noise_space_mapping @ self.data.T).T

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(mapped_centroids - mapped_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.assignments[cluster_assignment].append(self.data[i,:])
            self.cluster_space_assignments[cluster_assignment].append(mapped_data[i,:])
            self.noise_space_assignments[cluster_assignment].append(noise_data[i,:])

    def _update_transformation(self):
        # compute scatter matrix
        s_i = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in range(self.k):
            s_i += (utils.calculate_scatter(np.array(self.assignments[i])))

        # where we sub in randomized svd
        eigen_values, self.transform = utils.sorted_eig(s_i - self.s_d)
        self._get_M(eigen_values)

    def _get_M(self, eigen_values):
        self.m = len([i for i in eigen_values if i < -1e-10])


class SubKmeansRand(Kmeans):
    def __init__(self, k, data):
        super().__init__(k, data)
        self.m = int(np.sqrt(data.shape[1]))                           # cluster space dims
        self.transform = utils.init_transform(data.shape[1], m=self.m) # init transformation matrix
        self.s_d = utils.calculate_scatter(self.data)                  # compute scatter matrix S_D

    def _find_cluster_assignment(self):
        # re initialize clusters, as we are creating new assignments
        self.assignments = defaultdict(list)

        # map data to cluster space
        mapped_data = (self.transform.T @ self.data.T).T # (i by m) i being the datapoints

        # map centroids to cluster space
        mapped_centroids = (self.transform.T @ self.centroids.T).T  # (k by m) k being number of centroids

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(mapped_centroids - mapped_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.assignments[cluster_assignment].append(self.data[i,:])

    def _update_transformation(self):
        # compute scatter matrix
        s_i = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in range(self.k):
            s_i += (utils.calculate_scatter(np.array(self.assignments[i])))

        # where we sub in randomized svd
        eigen_values, self.transform = utils.sorted_eig(s_i - self.s_d, m=self.m)
        self._get_M(eigen_values)

    def _get_M(self, eigen_values):
        self.m = max(1, len([i for i in eigen_values if i < -1e-10]))


class PcaKmeans(Kmeans):
    def __init__(self, k, data):
        # run pca before clustering, and account for 90% of variance
        pca = PCA(n_components=0.9, svd_solver='full')
        data = pca.fit_transform(data)

        super().__init__(k, data)


    def _find_cluster_assignment(self):
        # re-initialize the clusters, as we are creating new assignments
        self.assignments = defaultdict(list)

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(self.centroids - self.data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.assignments[cluster_assignment].append(self.data[i, :])

    def _update_transformation(self):
        # pca kmeans doesn't iteratively update a transformation matrix
        pass


class LdaKmeans(Kmeans):
    def __init__(self, k, data):
        super().__init__(k, data)

        # set the subspace dimension
        self.d = k - 1

        self._lda = LinearDiscriminantAnalysis(n_components=self.d)

        # cluster label targets for LDA
        self.cluster_assignments = np.zeros((1,len(data)))

        # run pca to find intial subspace directions
        pca = PCA(n_components=self.d, svd_solver='full')
        pca.fit(self.data)
        self.transform = pca.components_

    def _find_cluster_assignment(self):
        # re-initialize the clusters, as we are creating new assignments
        self.assignments = defaultdict(list)
        self.cluster_assignments = np.zeros((1,len(data)))

        # transform the data
        transformed_data = self.data @ self.transform.T

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(self.centroids - transformed_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.cluster_assignments[i] = cluster_assignment
            self.assignments[cluster_assignment].append(self.data[i, :])

    def _update_transformation(self):
        # fit LDA using current class labels
        self._lda.fit(self.data, self.cluster_assignments)

        # get the LDA directions for the transformation
        self.transform = self._lda.coef_

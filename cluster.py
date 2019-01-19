import numpy as np
import utils
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import multiprocessing as mp
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

    def calc_cost(self):
        # implement cost function from https://doi.org/10.1145/3097983.3097989
        cost = 0

        # clustered subspace term
        for i in range(self.k):
            mapped_data = (self.pc.T @ self.transform.T @ np.vstack(self.assignments[i]).T).T
            mapped_centroids = (self.pc.T @ self.transform.T @ self.centroids[i].T).T
            cost += np.sum(np.linalg.norm(mapped_data - mapped_centroids, axis=1))

        # noise subspace term
        dataset_mean = np.mean(self.data, axis=0)
        noise_data = (self.pn.T @ self.transform.T @ self.data.T).T
        noise_mean = (self.pn.T @ self.transform.T @ dataset_mean.T).T
        cost += np.sum(np.linalg.norm(noise_data - noise_mean))

        return cost


class SubKmeansRand(Kmeans):
    def __init__(self, k, data):
        super().__init__(k, data)
        self.m = int(np.sqrt(data.shape[1]))                           # cluster space dims
        self.transform = utils.init_transform(data.shape[1], m=self.m) # init transformation matrix
        self.s_d = utils.calculate_scatter(self.data)                  # compute scatter matrix S_D
        self.s_i = []                                                  # scatter matrix S_i

    mapped_data = np.array([])
    mapped_centroids = np.array([])
    def _find_cluster_assignment(self):
        # re initialize clusters, as we are creating new assignments
        self.assignments = defaultdict(list)


        # map data to cluster space
        global mapped_data
        mapped_data = (self.transform.T @ self.data.T).T # (i by m) i being the datapoints

        # map centroids to cluster space
        global mapped_centroids
        mapped_centroids = (self.transform.T @ self.centroids.T).T  # (k by m) k being number of centroids

        # compute distances to centroids
        rows = np.array_split(np.arange(mapped_data.shape[0]), mp.cpu_count())
        lock = mp.Lock()
        with mp.Pool(initializer=self._init_lock, initargs=(lock,)) as pool:
            assignments = pool.map(self._compute_distances, rows)

        for dictionary in assignments:
            for k,v in dictionary.items():
                self.assignments[k].extend(v)


        # print(self.assignments)
        # for i in range(len(self.data)):
        #     dist = np.linalg.norm(mapped_centroids - mapped_data[i, :], axis=1)
        #     cluster_assignment = np.argmin(dist)
        #     self.assignments[cluster_assignment].append(self.data[i,:])

    def _compute_distances(self, rows):
        assignments = defaultdict(list)
        for row in rows:
            dist = np.linalg.norm(mapped_centroids - mapped_data[row, :], axis=1)
            cluster_assignment = np.argmin(dist)
            assignments[cluster_assignment].append(self.data[row,:])

        return assignments

    def _init_lock(self, l):
        global lock
        lock = l

    def _update_transformation(self):
        # compute scatter matrix
        s_i = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in range(self.k):
            s_i += (utils.calculate_scatter(np.array(self.assignments[i])))
        self.s_i = s_i

        # where we sub in randomized svd
        eigen_values, self.transform = utils.sorted_eig(s_i - self.s_d, m=self.m)
        self._get_M(eigen_values)

    def _get_M(self, eigen_values):
        self.m = max(1, len([i for i in eigen_values if i < -1e-10]))

    def calc_cost(self):
        scatter = self.s_i - self.s_d
        cost = np.matrix.trace(self.transform.T @ scatter @ self.transform) + \
               np.matrix.trace(self.transform.T @ self.s_d @ self.transform)
        return cost

class PcaKmeans(Kmeans):
    def __init__(self, k, data):
        super().__init__(k, data)

        # run pca before clustering, and account for 90% of variance
        pca = PCA(n_components=0.9, svd_solver='full')
        pca.fit(data)
        self.transform = pca.components_

    def _find_cluster_assignment(self):
        # re-initialize the clusters, as we are creating new assignments
        self.assignments = defaultdict(list)

        # transform the data
        transformed_data = self.data @ self.transform.T
        transformed_centroids = self.centroids @ self.transform.T

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(transformed_centroids - transformed_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.assignments[cluster_assignment].append(self.data[i, :])

    def _update_transformation(self):
        # pca kmeans doesn't iteratively update a transformation matrix
        pass

    def calc_cost(self):
        # calculate standard kmeans objective function in the projected subspace
        transformed_centroids = self.centroids @ self.transform.T

        cost = 0
        for i in range(self.k):
            transformed_data = np.vstack(self.assignments[i]) @ self.transform.T
            cost += np.sum(np.linalg.norm(transformed_data - transformed_centroids[i], axis=1))

        return cost


class LdaKmeans(Kmeans):
    def __init__(self, k, data):
        super().__init__(k, data)

        # set the subspace dimension
        self.d = k - 1

        self._lda = LinearDiscriminantAnalysis(n_components=self.d)

        # cluster label targets for LDA
        self.cluster_assignments = np.zeros(len(self.data))

        # run pca to find intial subspace directions
        pca = PCA(n_components=self.d, svd_solver='full')
        pca.fit(self.data)
        self.transform = pca.components_

    def _find_cluster_assignment(self):
        # re-initialize the clusters, as we are creating new assignments
        self.assignments = defaultdict(list)
        self.cluster_assignments = np.zeros(len(self.data))

        # transform the data
        transformed_data = self.data @ self.transform.T
        # XXX: the LdaKmeans paper uses U^T @ centroids
        transformed_centroids = self.centroids @ self.transform.T

        # compute distances to centroids
        for i in range(len(self.data)):
            dist = np.linalg.norm(transformed_centroids - transformed_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist)
            self.cluster_assignments[i] = cluster_assignment
            self.assignments[cluster_assignment].append(self.data[i, :])

    def _update_transformation(self):
        # fit LDA using current class labels
        self._lda.fit(self.data, self.cluster_assignments)

        # get the LDA directions for the transformation
        self.transform = self._lda.coef_

    def calc_cost(self):
        # calculate standard kmeans objective function in the projected subspace
        transformed_centroids = self.centroids @ self.transform.T

        cost = 0
        for i in range(self.k):
            transformed_data = np.vstack(self.assignments[i]) @ self.transform.T
            cost += np.sum(np.linalg.norm(transformed_data - transformed_centroids[i], axis=1))

        return cost

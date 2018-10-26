import numpy as np 
import utils 


class SubKmeans(object):
    def __init__(self, k, data):
        self.k = k                                           # number of clusters
        self.data = data 
        self.m = int(np.sqrt(data.shape[1]))                 # cluster space dims
        self.transform = utils.init_transform(data.shape[1]) # init transformation matrix

        # compute dataset mean -> used in scatter matrix
        self.data_mean = np.mean(data, axis=0)

        # computer scatter matrix S_D
        self.s_d = utils.calculate_scatter(self.data, self.data_mean) 

        # track cluster assignments with dict. cluster num is key
        self.assignments = dict.fromkeys(np.arange(k), [])

        # randomly assign points to clusters initially. 
        # Initial cluster centroids are chosen using random data points
        init_centroid_idx = np.random.choice(len(data), k, replace=False)
        self.centroids = self.data[init_centroid_idx, :]

    def _update_centroids(self):
        for k, v in self.assignments.items():
            pts = np.vstack(v)
            centroid = np.mean(pts, axis=0)
            self.centroids[k,:] = centroid

    def _find_cluster_assignment(self):
        # re initialize clusters, as we are creating new assignments
        self.assignments = dict.fromkeys(np.arange(self.k), [])

        # calculate the cluster space mapping 
        pc = utils.calc_pc(self.data.shape[1], self.m)    # calc the projection matrix
        cluster_space_mapping = pc.T @ self.transform.T   # calc the cluster space mapping

        # map data to cluster space
        mapped_data = (cluster_space_mapping @ self.data.T).T # (i by m) i being the datapoints

        # map centroids to cluster space
        mapped_centroids = (cluster_space_mapping @ self.centroids.T).T  # (k by m) k being number of centroids

        # compute distances to centroids 
        for i in range(len(self.data)):
            dist = np.linalg.norm(mapped_centroids - mapped_data[i, :], axis=1)
            cluster_assignment = np.argmin(dist) 
            self.assignments[cluster_assignment].append(self.data[i,:])


    def _update_transformation(self):
        # compute scatter matrix
        s_i = np.zeros((self.data.shape[1], self.data.shape[1]))
        for k,v in self.assignments.items():
            for i in v:
                s_i += (i - self.centroids[k, :]) @ (i - self.centroids[k, :]).T
        
        V, eigen_values, eigen_vectors = utils.eigen_decomp(s_i, self.s_d)
        self.transform = V 

        self._get_M(eigen_values)

    def _get_M(self, eigen_values):
        self.m = len([i for i in eigen_values if i < -1e-10])


    
    
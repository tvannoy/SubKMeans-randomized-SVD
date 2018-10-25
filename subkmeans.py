import numpy as np 


class SubKmeans(object):
    def __init__(self, k, data):
        self.k = k
        self.data = data 
        self.m = np.sqrt(data.shape[1]) # sqrt of number of features 

        # compute dataset mean -> used in scatter matrix
        self.data_mean = np.mean(data, axis=0)

        # computer scatter matrix S_D
        self.s_d = (self.data - self.data_mean) @ (self.data - self.data_mean).T 

        # track cluster assignments with dict. cluster num is key
        self.assignments = dict.fromkeys(np.arange(k), [])

        # randomly assign points to clusters initially. 
        # Initial cluster are chosen using random data points
        init_centroid_idx = np.random.choice(len(data), k, replace=False)
        self.centroids = self.data[init_centroid_idx, :]


    def _update_centroids(self):
        pass 

    def _find_cluster_assignment(self):
        pass 

    def _update_transformation(self):
        pass 

    def _get_M(self):
        pass 

    def _get_costs(self): 
        pass 

    
    
import numpy as np
import utils
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

class Kmeans(object):
    def __init__(self, k, data):
        self.k = k                            # number of clusters
        self.data = data                      # dataset
        self.assignments = defaultdict(list)  # track cluster assignments with dict. cluster num is key

        # randomly assign points to clusters initially.
        # Initial cluster centroids are chosen using random data points
        init_centroid_idx = np.random.choice(len(data), k, replace=False)
        self.centroids = self.data[init_centroid_idx, :]

    def run(self, max_iter=1000, tol=0.99):
        self._find_cluster_assignment()
        self._update_centroids()
        self._update_transformation()

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
            self._update_transformation()

            cur_labels = []
            for k,v in self.assignments.items():
                cur_labels += list(k * np.ones(len(v)))

            nmi = normalized_mutual_info_score(prev_labels, cur_labels, average_method='arithmetic')
            n += 1
        print("NMI: {}, n_iter: {}".format(nmi, n))


    def _update_centroids(self):
        for k, v in self.assignments.items():
            pts = np.vstack(v)
            centroid = np.mean(pts, axis=0)
            self.centroids[k,:] = centroid


    def _find_cluster_assignment(self):
        raise NotImplementedError("Please implement this method.")


    def _update_transformation(self):
        raise NotImplementedError("Please implement this method.")

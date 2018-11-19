#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import scale 
from time import perf_counter, strftime, gmtime

import cluster

def data_size_test(algorithm, sets):
    median_runtimes = []
    nmi = [] 
    for data, labels in sets:
        print("\nDimensionality: {}\n".format(data.shape[1]))

        runtimes = []
        loc_nmi = []

        # run algorithm 20 times and record the median runtime
        for i in range(20):
            alg = algorithm(3, data)

            t0 = perf_counter()
            alg.run()
            t1 = perf_counter()

            runtimes.append(t1 - t0)

            cur_labels = []
            for k,v in alg.assignments.items():
                cur_labels += list(k * np.ones(len(v)))
            loc_nmi.append(normalized_mutual_info_score(sorted(labels), sorted(cur_labels)))

        median_runtime = np.median(runtimes)
        median_runtimes.append(median_runtime)
        nmi.append(np.mean(loc_nmi))
        # print("median runtime: {}".format(median_runtime))

    print("median runtimes: {}".format(median_runtimes))
    print("NMI: {}".format(nmi))

    return (dim_sizes, median_runtimes, nmi)

if __name__ == '__main__':
    algorithms = (cluster.SubKmeansRand, cluster.SubKmeans, cluster.PcaKmeans, cluster.LdaKmeans)

    keys = [alg.__name__ for alg in algorithms]
    results = dict.fromkeys(keys)

    dim_sizes = np.logspace(2, 4, 8, dtype=np.int)
    sets = []
    for d in dim_sizes:
        # create synthetic dataset
        # data, labels = make_classification(n_samples=1000, n_features=d,
        #     n_informative=10, n_classes=3, n_redundant=0, n_clusters_per_class=1)
        data, labels = make_blobs(n_samples=1000, n_features=d, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
        data = scale(data) 
        sets.append((data,labels))

    for alg in algorithms:
        print("running dimensionality test on {}".format(alg.__name__))
        dim_sizes, median_runtimes, nmi = data_size_test(alg, sets)
        results[alg.__name__] = (dim_sizes, median_runtimes, nmi)

        # save results
        results_dir = os.path.join(os.getcwd(), "Results")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        t = strftime("%H_%M_%S", gmtime())
        filename = os.path.join(results_dir, "dim_results_" + alg.__name__ + "_" + t + ".csv")
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['dim_size', 'median_runtime', 'NMI'])
            for size, runtime, nmi in zip(dim_sizes, median_runtimes, nmi):
                writer.writerow([size, runtime, nmi])


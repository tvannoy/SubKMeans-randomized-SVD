#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from sklearn.datasets import make_classification
from time import perf_counter, strftime, gmtime

import cluster

def data_size_test(algorithm):

    sample_sizes = np.logspace(2, 4, 20, dtype=np.int)

    median_runtimes = []
    for n_samples in sample_sizes:
        # create synthetic dataset
        data, labels = make_classification(n_samples=n_samples, n_features=20,
            n_informative=2, n_classes=3, n_redundant=0, n_clusters_per_class=1)

        runtimes = []

        # run algorithm 20 times and record the median runtime
        for i in range(20):
            alg = algorithm(3, data)

            t0 = perf_counter()
            alg.run()
            t1 = perf_counter()

            runtimes.append(t1 - t0)

        median_runtime = np.median(runtimes)
        median_runtimes.append(median_runtime)
        # print("median runtime: {}".format(median_runtime))

    print("median runtimes: {}".format(median_runtimes))

    return (sample_sizes, median_runtimes)

if __name__ == '__main__':
    algorithms = (cluster.SubKmeansRand, cluster.SubKmeans, cluster.PcaKmeans, cluster.LdaKmeans)

    keys = [alg.__name__ for alg in algorithms]
    results = dict.fromkeys(keys)

    for alg in algorithms:
        print("running data size test on {}".format(alg.__name__))
        sample_sizes, median_runtimes = data_size_test(alg)
        results[alg.__name__] = (sample_sizes, median_runtimes)

        # save results
        results_dir = os.path.join(os.getcwd(), "Results")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        t = strftime("%H_%M_%S", gmtime())
        filename = os.path.join(results_dir, "runtime_results_" + t + ".csv")
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_size', 'median_runtime'])
            for size, runtime in zip(sample_sizes, median_runtimes):
                writer.writerow([size, runtime])

    for alg in keys:
        plt.loglog(results[alg][0], results[alg][1] ,'-o')
    plt.legend(keys)
    plt.xlabel('sample size')
    plt.ylabel('median runtime [s]')
    plt.show()

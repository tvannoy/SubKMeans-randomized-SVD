#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from sklearn.datasets import make_classification
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import scale
from time import perf_counter, strftime, gmtime

import cluster

def data_size_test(algorithm, sets):
    average_runtimes = []
    nmi = []
    for data, labels in sets:
        print("Datasize: {}".format(len(data)))
        runtimes = []
        loc_nmi = []

        # run algorithm 20 times and record the average runtime
        for i in range(20):
            alg = algorithm(3, data)

            t0 = perf_counter()
            alg.run()
            t1 = perf_counter()

            runtimes.append(t1 - t0)

            cur_labels = []
            for k,v in alg.assignments.items():
                cur_labels += list(k * np.ones(len(v)))
            loc_nmi.append(normalized_mutual_info_score(labels, cur_labels))

        average_runtime = np.mean(runtimes)
        runtime_dev = np.std(runtimes)
        average_runtimes.append((average_runtime, runtime_dev))
        nmi.append(np.mean(loc_nmi))
        print("\naverage runtime: {}\n".format(average_runtime))

    print("average runtimes: {}".format(average_runtimes))
    print("NMI: {}".format(nmi))

    return (sample_sizes, average_runtimes, nmi)

if __name__ == '__main__':
    algorithms = (cluster.SubKmeansRand, cluster.SubKmeans, cluster.PcaKmeans, cluster.LdaKmeans)

    keys = [alg.__name__ for alg in algorithms]
    results = dict.fromkeys(keys)

    sample_sizes = np.logspace(2, 5, 8, dtype=np.int)
    sets = []
    for n_samples in sample_sizes:
        # create synthetic dataset
        data, labels = make_classification(n_samples=n_samples, n_features=100,
                                           n_informative=20, n_classes=3, n_redundant=0, n_clusters_per_class=1)
        data = scale(data)
        sets.append((data, labels))

    for alg in algorithms:
        print("running data size test on {}".format(alg.__name__))
        sample_sizes, average_runtimes, nmi = data_size_test(alg, sets)
        results[alg.__name__] = (sample_sizes, average_runtimes, nmi)

        # save results
        results_dir = os.path.join(os.getcwd(), "Results")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        t = strftime("%H_%M_%S", gmtime())
        filename = os.path.join(results_dir, "runtime_results_" + alg.__name__ + "_" +  t + ".csv")
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_size', 'average_runtime', 'StDev', 'NMI'])
            for size, runtime, nmi in zip(sample_sizes, average_runtimes, nmi):
                writer.writerow([size, runtime[0], runtime[1], nmi])

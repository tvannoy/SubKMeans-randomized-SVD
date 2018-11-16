#!/usr/bin/python

import pickle
import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import datasets
from time import gmtime, strftime

import cluster


def cluster_test(algorithm, data, labels):
    data = preprocessing.scale(data)

    results = []
    costs = []

    # run each algorithm 40 times
    for i in range(40):
        alg = algorithm(len(set(labels)), data) # choose your algorithm
        alg.run()

        # calculate cluster labels
        cur_labels = []
        for k,v in alg.assignments.items():
            cur_labels += list(k * np.ones(len(v)))
        print(labels)
        print(cur_labels)

        # compute metrics
        nmi = normalized_mutual_info_score(labels, cur_labels, average_method='arithmetic')

        # store result as tuple
        results.append(nmi)

        # compute cost
        cost = alg.calc_cost()
        costs.append(cost)

    # sort costs in ascending order
    args = np.argsort(costs)

    # take 20 lowest costs
    results = [results[i] for i in args[0:20]]
    print(results)

    return np.median(results)

if __name__ == "__main__":
    # load in the Plane dataset
    plane = np.genfromtxt('datasets/Plane/Plane_combined', delimiter=',')
    data_name = 'plane'
    labels = plane[:,0] - 1
    data = plane[:,1:]

    # load in the OliveOil dataset
    # oliveoil = np.genfromtxt('datasets/OliveOil/OliveOil_combined', delimiter=',')
    # data_name = 'OliveOil'
    # labels = oliveoil[:,0]
    # data = oliveoil[:,1:]

    # algorithms = (cluster.SubKmeansRand, cluster.SubKmeans, cluster.PcaKmeans, cluster.LdaKmeans)
    algorithms = (cluster.SubKmeans,)
    keys = [alg.__name__ for alg in algorithms]
    results = dict.fromkeys(keys)

    for alg in algorithms:
        print("running {} on {}".format(alg.__name__, data_name))
        results[alg.__name__] = cluster_test(alg, data, labels)


    print(results)
    # save results
    # results_dir = os.path.join(os.getcwd(), "Results")
    # if not os.path.exists(results_dir):
    #     os.mkdir(results_dir)
    #
    # t = strftime("%H_%M_%S", gmtime())
    # filename = os.path.join(results_dir, "results_" + t + ".p")
    # with open(filename, 'w') as f:

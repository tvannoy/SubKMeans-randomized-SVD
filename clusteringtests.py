#!/usr/bin/python

import pickle
import os
import  csv
import scipy.io
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import datasets
from time import gmtime, strftime, perf_counter
from collections import defaultdict
from itertools import count
from functools import partial

import cluster


def cluster_test(algorithm, data, labels):
    data = preprocessing.scale(data)

    results = []
    times = []
    costs = []

    # run each algorithm 40 times
    for i in range(40):
        alg = algorithm(len(set(labels)), data)
        t0 = perf_counter()
        alg.run()
        t1 = perf_counter()

        # store runtime
        times.append(t1 - t0)

        # calculate cluster labels
        cur_labels = []
        for k,v in alg.assignments.items():
            cur_labels += list(k * np.ones(len(v)))
        # print(labels)
        # print(cur_labels)

        # compute metrics
        nmi = normalized_mutual_info_score(sorted(labels), sorted(cur_labels), average_method='arithmetic')

        # store result as tuple
        results.append(nmi)

        # compute cost
        cost = alg.calc_cost()
        costs.append(cost)

    # sort costs in ascending order
    args = np.argsort(costs)

    # take 20 lowest costs
    results = [results[i] for i in args[0:20]]
    times = [times[i] for i in args[0:20]]
    # print(results)

    return (np.median(results), np.median(times))

if __name__ == "__main__":
    # load in the Plane dataset
    # plane = np.genfromtxt('datasets/Plane/Plane_combined', delimiter=',')
    # data_name = 'plane'
    # labels = plane[:,0] - 1 # subtract one because our class labels start at 0 and the dataset's labels start at 1.
    # data = plane[:,1:]

    # load in the OliveOil dataset
    # oliveoil = np.genfromtxt('datasets/OliveOil/OliveOil_combined', delimiter=',')
    # data_name = 'OliveOil'
    # labels = oliveoil[:,0] - 1
    # data = oliveoil[:,1:]

    # load in the StarLightCurves dataset
    starlight = np.genfromtxt('datasets/StarLightCurves/StarLightCurves_combined', delimiter=',')
    data_name = 'StarLightCurves'
    labels = starlight[:,0] - 1
    data = starlight[:,1:]

    # load in the Symbols dataset
    # symbols = np.genfromtxt('datasets/Symbols/Symbols_combined', delimiter=',')
    # data_name = 'Symbols'
    # labels = symbols[:,0] - 1
    # data = symbols[:,1:]

    # load in the DrivFace dataset
    # matfile = scipy.io.loadmat('datasets/DrivFace/DrivFace.mat') # load in matlab data file
    # drivFace = matfile['drivFaceD'][0,0] # grab the struct
    # data = drivFace['data']
    # labels = drivFace['nlab'][:,0]
    # data_name = 'drivFace'

    # load in the RNASeq dataset
    # data = np.genfromtxt('datasets/TCGA-PANCAN-HiSeq-801x20531/data.csv', delimiter=',', skip_header=True)
    # data = data[:,1:] # get rid of sample_# column
    # labels = np.genfromtxt('datasets/TCGA-PANCAN-HiSeq-801x20531/labels.csv', delimiter=',', skip_header=True, dtype=str)
    # labels = labels[:,1] # rid of smaple_# column
    # # convert label strings into numbers (https://stackoverflow.com/questions/17152468/python-enumerate-a-list-of-string-keys-into-ints)
    # label_to_num = defaultdict(partial(next, count(0)))
    # labels = np.array([label_to_num[label] for label in labels])
    # data_name = 'RNASeq'

    # algorithms = (cluster.SubKmeansRand, cluster.SubKmeans, cluster.PcaKmeans, cluster.LdaKmeans)
    algorithms = (cluster.SubKmeans, cluster.SubKmeansRand)
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
    # filename = os.path.join(results_dir, "clustering_" + data_name + "_" + t + ".csv")
    # with open(filename, 'w') as f:
    #     nmi, runtime = zip(*results.values())
    #     writer = csv.writer(f)
    #     writer.writerow(results.keys())
    #     writer.writerow(nmi)
    #     writer.writerow(runtime)

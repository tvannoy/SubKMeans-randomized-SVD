import pickle 
import os 
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import datasets 
from time import gmtime, strftime

import cluster

if __name__ == "__main__":
    # load dataset
    # with open("file.p", 'rb') as f:
    #     tup = pickle.load(f)
    # labels, data = tup

    dat = datasets.load_wine() 
    data = dat.data 
    labels = dat.target 
    data = preprocessing.scale(data)

    results = []
    costs = []

    # run each algorithm 40 times
    for i in range(40):
        alg = cluster.SubKmeansRand(3, data) # choose your algorithm
        alg.run() 

        # calculate cluster labels
        cur_labels = []
        for k,v in alg.assignments.items():
            cur_labels += list(k * np.ones(len(v)))

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

    # save results
    results_dir = os.path.join(os.getcwd(), "Results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    t = strftime("%H_%M_%S", gmtime())
    filename = os.path.join(results_dir, "results_" + t + ".p")
    with open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)






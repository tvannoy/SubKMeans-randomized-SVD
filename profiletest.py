#!/usr/bin/python

import pickle
import os
import csv
import sys
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




if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
    except:
        print('give a dataset name')
        exit()

    if dataset.lower() == 'plane':
        # load in the Plane dataset
        plane = np.genfromtxt('datasets/Plane/Plane_combined', delimiter=',')
        data_name = 'plane'
        labels = plane[:,0] - 1 # subtract one because our class labels start at 0 and the dataset's labels start at 1.
        data = plane[:,1:]
    elif dataset.lower() == 'oliveoil':
        # load in the OliveOil dataset
        oliveoil = np.genfromtxt('datasets/OliveOil/OliveOil_combined', delimiter=',')
        data_name = 'OliveOil'
        labels = oliveoil[:,0] - 1
        data = oliveoil[:,1:]
    elif dataset.lower() == 'starlightcurves':
        starlight = np.genfromtxt('datasets/StarLightCurves/StarLightCurves_combined', delimiter=',')
        data_name = 'StarLightCurves'
        labels = starlight[:,0] - 1
        data = starlight[:,1:]
    elif dataset.lower() == 'symbols':
        # load in the Symbols dataset
        symbols = np.genfromtxt('datasets/Symbols/Symbols_combined', delimiter=',')
        data_name = 'Symbols'
        labels = symbols[:,0] - 1
        data = symbols[:,1:]
    elif dataset.lower() == 'drivface':
        # load in the DrivFace dataset
        matfile = scipy.io.loadmat('datasets/DrivFace/DrivFace.mat') # load in matlab data file
        drivFace = matfile['drivFaceD'][0,0] # grab the struct
        data = drivFace['data']
        labels = drivFace['nlab'][:,0]
        data_name = 'drivFace'
    elif dataset.lower() == 'rnaseq':
        # load in the RNASeq dataset
        data = np.genfromtxt('datasets/TCGA-PANCAN-HiSeq-801x20531/data.csv', delimiter=',', skip_header=True)
        data = data[:,1:] # get rid of sample_# column
        labels = np.genfromtxt('datasets/TCGA-PANCAN-HiSeq-801x20531/labels.csv', delimiter=',', skip_header=True, dtype=str)
        labels = labels[:,1] # rid of smaple_# column
        # convert label strings into numbers (https://stackoverflow.com/questions/17152468/python-enumerate-a-list-of-string-keys-into-ints)
        label_to_num = defaultdict(partial(next, count(0)))
        labels = np.array([label_to_num[label] for label in labels])
        data_name = 'RNASeq'
    elif dataset.lower() == 'random':
        data, labels = make_classification(n_samples=20000, n_features=20000,
            n_informative=20, n_classes=11, n_redundant=0, n_clusters_per_class=1)
    else:
        print('invalid dataset name')
        exit()

    data = preprocessing.scale(data)
    alg = cluster.SubKmeansRand(len(set(labels)), data)
    alg.run()

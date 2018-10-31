from sklearn import preprocessing
import subkmeans
import pickle 
import time 

if __name__ == "__main__":
    with open("Data/postures.p", 'rb') as f:
        data = pickle.load(f)

    X = preprocessing.scale(data)
    kmeans = subkmeans.SubKmeans(5, X)   

    t0 = time.time()
    kmeans.run(randomized=True)
    t1 = time.time()

    print("Elapsed Time: {}".format(t1-t0))
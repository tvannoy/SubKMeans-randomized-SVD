import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import perf_counter

import utils


def calc_scatter_sequential(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    S = centered_data.T @ centered_data
    return S


if __name__ == '__main__':
    avg_runtimes_sequential = []
    avg_runtimes_parallel = []

    dim_sizes = 3*np.logspace(2, 4, 8, dtype=np.int)
    for d in dim_sizes:
        runtimes_sequential = []
        runtimes_parallel = []

        for i in range(4):
            data = np.random.rand(5000, d)

            t0 = perf_counter()
            calc_scatter_sequential(data)
            t1 = perf_counter()
            runtimes_sequential.append(t1-t0)

            t0 = perf_counter()
            utils.calculate_scatter(data)
            t1 = perf_counter()
            runtimes_parallel.append(t1-t0)

        avg_runtimes_sequential.append(np.average(runtimes_sequential))
        avg_runtimes_parallel.append(np.average(runtimes_parallel))

    print(avg_runtimes_sequential)
    print(avg_runtimes_parallel)
    
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    plt.loglog(dim_sizes, avg_runtimes_sequential, label='sequential', marker='o')
    plt.loglog(dim_sizes, avg_runtimes_parallel, label='parallel', marker='s')
    plt.xlabel('No. of Dimensions')
    plt.ylabel('Average Runtime [sec]')
    plt.legend()
    plt.savefig('scatter_matrix_performance.png', dpi=600, format='png')
    # plt.show()

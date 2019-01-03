import unittest
import numpy as np
import utils

class TestScatterMatrixComputation(unittest.TestCase):
    def test_random(self):
        # create data
        data = np.random.rand(500, 300)
        mean = np.mean(data, axis=0)

        # calculate scatter matrix sequentially
        S1 = (data - mean).T @ (data - mean)

        # calculate scatter matrix in parallel
        S2 = utils.calculate_scatter(data)

        # make sure all elements are the same
        self.assertTrue((S1 == S2).all())

    def test_ones(self):
        # create data
        data = np.ones([500, 300])
        mean = np.mean(data, axis=0)

        # calculate scatter matrix sequentially
        S1 = (data - mean).T @ (data - mean)

        # calculate scatter matrix in parallel
        S2 = utils.calculate_scatter(data)

        # make sure all elements are the same
        self.assertTrue((S1 == S2).all())


if __name__ == '__main__':
    unittest.main()



import unittest

from keras_nn_test import runTrainTest
from keras_nn_score_test import runScoreTest

class Tests(unittest.TestCase):

    def test_classification(self):
        self.assertTrue(runTrainTest("classification"))
        self.assertTrue(runScoreTest("classification"))

    def test_regression(self):
        self.assertTrue(runTrainTest("regression"))
        self.assertTrue(runScoreTest("regression"))

    def test_autoregression(self):
        self.assertTrue(runTrainTest("autoregression"))
        self.assertTrue(runScoreTest("autoregression"))

    def test_text_classification(self):
        self.assertTrue(runTrainTest("text_classification"))
        self.assertTrue(runScoreTest("text_classification"))

    def __test_time_series(self):
        self.assertTrue(runTrainTest("time_series"))
        self.assertTrue(runScoreTest("time_series"))

    def test_unsupervised(self):
        self.assertTrue(runTrainTest("unsupervised"))
        self.assertTrue(runScoreTest("unsupervised"))

if __name__ == '__main__':
    unittest.main(verbosity=2)


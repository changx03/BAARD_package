from sklearn import datasets
from sklearn import svm
import numpy as np
import os
import sys
import unittest
from baard import ml_base_class as mlbc

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BasicDetector(mlbc.BAARD):

    def predict(self, X):
        return self.model.predict(X)

    def encoding(self, X):
        return X

class BasicModel():

    def predict(self, X):
        return np.ones(X.shape,dtype=int)

class Testmlbc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        dummy_model = BasicModel()
        cls.detector1 = BasicDetector(dummy_model)

    def setUp(self):
        pass

    def test_predict(self):
        X1 = [[1,2,3,4],
              [5,6,7,8]]
        X2 = [0.]
        X3 = []
        assert(self.detector1.predict(X1) == [[1,1,1,1],[1,1,1,1]])
        assert(self.detector1.predict(X2) == [1])
        assert(self.detector1.predict(X3) == [])

    def test_detect(self):
        pass

    def test_encoding(self):
        pass

if __name__ == '__main__':
    unittest.main()

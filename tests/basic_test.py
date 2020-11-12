from sklearn import datasets
from sklearn import svm
import numpy as np
import os
import sys
import unittest
from BAARD import ml_base_class as mlbc

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IrisSVCModel(mlbc.BAARD):

    def predict(self, X):
        return self.model.predict(X)

class Testmlbc(unittest.TestCase):
    
    def test_init(self):

        exception_raised = False
        try:
            bc = IrisSVCModel()
        except TypeError:
            exception_raised = True

        self.assertTrue(exception_raised)

    def test_predict(self):
        
        """ This code is from: https://scikit-learn.org/stable/tutorial/basic/tutorial.html """
        iris = datasets.load_iris()
        svm_model = svm.SVC(gamma=0.001, C=100.0)
        svm_model.fit(iris.data[:-1], iris.target[:-1])

        bc = IrisSVCModel(svm_model)
        self.assertTrue(type(bc.model) is svm.SVC)

        data = {'sepal_length': 1.0,
            'sepal_width': 1.0,
            'petal_length': 1.0,
            'petal_width': 1.0}
        X = np.array([data["sepal_length"], data["sepal_width"], \
            data["petal_length"], data["petal_width"]]).reshape(1, -1)

        y_hat = bc.predict(X=X)[0]
        targets = ['setosa', 'versicolor', 'virginica']
        species = targets[y_hat]
        self.assertTrue(species == 'setosa')

    def test_detect(self):
        pass

    def test_encoding(self):
        pass

if __name__ == '__main__':
    unittest.main()

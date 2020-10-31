from sklearn import datasets
from sklearn import svm
import numpy as np
import os
import sys

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_base_class import BAARD

def main():
    """ This code is from: https://scikit-learn.org/stable/tutorial/basic/tutorial.html """
    iris = datasets.load_iris()
    svm_model = svm.SVC(gamma=0.001, C=100.0)
    svm_model.fit(iris.data[:-1], iris.target[:-1])

    data = {'sepal_length': 1.0,
        'sepal_width': 1.0,
        'petal_length': 1.0,
        'petal_width': 1.0}
    X = np.array([data["sepal_length"], data["sepal_width"], \
        data["petal_length"], data["petal_width"]]).reshape(1, -1)
    bc = BAARD(svm_model)
    y_hat = bc.predict(X=X)[0]
    targets = ['setosa', 'versicolor', 'virginica']
    species = targets[y_hat]
    print({"species":species})

if __name__ == "__main__":
    main()
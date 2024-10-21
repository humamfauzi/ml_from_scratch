import unittest
import statistics
import math
from collections import Counter

def euclidean(X, Y):
    if len(X) != len(Y):
        raise ValueError(f"the length of train data {len(X)}, and target data {len(Y)} are not the same")
    total = len(X)
    ssum = sum([(X[index] - Y[index]) ** 2 for index in range(len(X))])
    return math.sqrt(ssum)

class KNNClassification:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.__is_fitted = False
        self.__X = None
        self.__y = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"the length of train data {len(X)}, and target data {len(Y)} are not the same")
        self.__X = X
        self.__y = y
        self.__is_fitted = True
        self.__fdistance = euclidean

    def _generating_k_nearest(self, input_value, smallest_distance):
        for index, reference_value in enumerate(self.__X):
            distance = self.__fdistance(input_value, reference_value)
            if len(smallest_distance) != self.n_neighbors:
                smallest_distance.append({"index": index, "distance": distance})
            if len(smallest_distance) == self.n_neighbors and smallest_distance[-1]["distance"] > distance:
                smallest_distance[self.n_neighbors-1] = {"index": index, "distance": distance}
                # keeping the largest value in the bottom when there is insertion
                smallest_distance.sort(key=lambda x: x["distance"])


    def predict(self, X):
        if not self.__is_fitted:
            raise ValueError(f"please call fit first")
        prediction = []
        for i in X:
            smallest_distance = []
            if len(i) != len(self.__X[0]):
                raise ValueError(f"different data dimension, input {len(i)}, trained {len(self.__X[0])}")
            self._generating_k_nearest(i, smallest_distance)

            y_value = [self.__y[i["index"]] for i in smallest_distance]
            counter = Counter(y_value)
            conc = max(counter, key=counter.get)
            prediction.append(conc)
        return prediction

class KNNRegressor:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.__is_fitted = False
        self.__X = None
        self.__y = None
        self._policy = "average"

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"the length of train data {len(X)}, and target data {len(y)} are not the same")
        self.__X = X
        self.__y = y
        self.__is_fitted = True
        self.__fdistance = euclidean
        return 

    def _generating_k_nearest(self, input_value, smallest_distance):
        for index, reference_value in enumerate(self.__X):
            distance = self.__fdistance(input_value, reference_value)
            if len(smallest_distance) != self.n_neighbors:
                smallest_distance.append({"index": index, "distance": distance})
            if len(smallest_distance) == self.n_neighbors and smallest_distance[-1]["distance"] > distance:
                smallest_distance[self.n_neighbors-1] = {"index": index, "distance": distance}
                # keeping the largest value in the bottom when there is insertion
                smallest_distance.sort(key=lambda x: x["distance"])

    def predict(self, X):
        if not self.__is_fitted:
            raise ValueError(f"please call fit first")
        prediction = []
        for i in X:
            smallest_distance = []
            if len(i) != len(self.__X[0]):
                raise ValueError(f"different data dimension, input {len(i),}, trained {len(self.__X[0])}")
            self._generating_k_nearest(i, smallest_distance)
            y_value = [self.__y[i["index"]] for i in smallest_distance]
            prediction.append(statistics.mean(y_value))
        return prediction


class TestKNN(unittest.TestCase):
    def test_classification(self):
        p1, y1 = [1,1], 1
        p2, y2 = [-1, 1], 1
        p3, y3 = [-1, -1], 2
        p4, y4 = [1, -1], 2

        X = [p1, p2, p3, p4]
        y = [y1, y2, y3, y4]

        knn = KNNClassification(n_neighbors=2)
        knn.fit(X, y)
        p5, y5  = [0,2], 1
        p6, y6  = [0,-3], 2
        X_pred = [p5, p6]
        yp1, yp2 = knn.predict(X_pred)
        self.assertEqual(yp1, y5)
        self.assertEqual(yp2, y6)

    def test_regressor(self):
        p1, y1 = [1,1], 10
        p2, y2 = [-1, 1], 20
        p3, y3 = [-1, -1], -10
        p4, y4 = [1, -1], -20

        X = [p1, p2, p3, p4]
        y = [y1, y2, y3, y4]

        knn = KNNRegressor(n_neighbors=2)
        knn.fit(X, y)
        p5, y5  = [0,2], 15
        p6, y6  = [0,-3], -15
        X_pred = [p5, p6]
        yp1, yp2 = knn.predict(X_pred)
        self.assertEqual(yp1, y5)
        self.assertEqual(yp2, y6)

if __name__ == "__main__":
    unittest.main()

from point import Point
import unittest

class Distance:
    # this should calcuate usingg actual class without any instantiation
    def __init__(self):
        return

    def euclidean(self, p1: Point, p2: Point) -> float:
        if len(p1) < 2:
            raise ValueError("wrong p1; Euclidean require two value")
        if len(p2) < 2:
            raise ValueError("wrong p2; Euclidean require two value")
        squared1 = (p1[0] - p2[0]) ** 2
        squared2 = (p1[1] - p2[1]) ** 2
        return (squared1 + squared2) ** 0.5

    def minkowski(self, p1: Point, p2: Point) -> float:
        if len(p1.data) != len(p2.data):
            raise ValueError("minkowski require both point has same length")
        total = len(p1.data)
        return np.sum([ (p1[i] - p2[i]) ** total for i in range(total)]) ** 1/total

class TestDistance(unittest.TestCase):
    def test_euclidean(self):
        dist = Distance()
        p1 = [1,1]
        p2 = [2]
        with self.assertRaises(ValueError):
            dist.euclidean(p1, p2)

if __name__ == "__main__":
    unittest.main()

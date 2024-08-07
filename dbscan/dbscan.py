import unittest
from point import Point
from typing import List

class DBSCAN:
    def __init__(self, dataset, distance_func, min_point, epsilon):
        self.dataset = [Point(d) for d in dataset]
        self.epsilon = epsilon
        self.distance_func = distance_func
        self.min_point = min_point
        
    # definition 1
    def epsilon_neighbor(self, reference: Point):
        total = []
        for point in self.dataset:
            if self.distance_func(reference, point) <= self.epsilon:
                reference.add_neighbor(point)
        return self

    # definition 2
    def is_directly_density_reachable(self, questioned_point: Point, reference_point: Point) -> bool:
        return reference_point.has_neighbor(questioned_point) and reference_point.neighbor_size() >= self.min_point

    # definition 3
    def is_density_reachable(self, questioned_point: Point, reference_point: Point, already_visited: List[Point]) -> bool:
        # this should be recusively find all possible neighbor. the exhautive list would taken a long time.
        # we would optimise this function later.
        # there are also possibilities of densitiy reachable create a circular function
        for neighbor in reference_point.get_neighbors:
            if questioned_point in neighbor.get_neighbors():
                return True
            if is_density_reachable(questioned_point, neighbor, already_visited):
                return True
            already_visited.append(neighbor)
        return False


class TestDBSCAN(unittest.TestCase):
    def test_epsilon(self):
        def euclidean(p1, p2):
            return ((p1.data[0] - p2.data[0]) ** 2 + (p1.data[1] - p2.data[1]) ** 2 ) ** .5
        p1 = [1,2]
        p2 = [3,4]
        p3 = [9,12]
        db = DBSCAN([p1, p2, p3], euclidean, 1, 5)
        p4 = Point([0, 0])
        db.epsilon_neighbor(p4)
        self.assertEqual(p4.neighbor_size(), 2)

    def test_is_directly_density_reachable(self):
        db = DBSCAN([], euclidean, 1, 5)
        p1 = [1,2]

        db.is_directly_density_reachable(
        

if __name__ == "__main__":
    unittest.main()

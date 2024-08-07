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
    def run_epsilon_neighbor(self, reference: Point):
        for point in self.dataset:
            if point.has_neighbor(reference):
                continue
            if self.distance_func(reference, point) <= self.epsilon:
                # adding here should not be a problem
                # since point is a class therefore
                # it always called as a reference instead of copying
                reference.add_neighbor(point)
                point.add_neighbor(reference)
        return self

    def run_all_epsilon_neighbor(self):
        for point in self.dataset:
            self.run_epsilon_neighbor(point)
        return self

    # definition 2
    def is_directly_density_reachable(self, questioned_point: Point, reference_point: Point) -> bool:
        is_neighbor = reference_point.has_neighbor(questioned_point)
        has_min_point = reference_point.neighbor_size() >= self.min_point
        print("<>>>>>>", is_neighbor, has_min_point)
        return is_neighbor and has_min_point

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


def euclidean(p1, p2):
    return ((p1.data[0] - p2.data[0]) ** 2 + (p1.data[1] - p2.data[1]) ** 2 ) ** .5

class TestDBSCAN(unittest.TestCase):
    def test_epsilon(self):
        p1 = [1,2]
        p2 = [3,4]
        p3 = [9,12]
        db = DBSCAN([p1, p2, p3], euclidean, 1, 5)
        p4 = Point([0, 0])
        db.run_epsilon_neighbor(p4)
        self.assertEqual(p4.neighbor_size(), 2)

    def test_is_directly_density_reachable(self):
        p1 = [1,1]
        p2 = [-1,1]
        p3 = [1,-1]
        p4 = [-1,-1]
        central = [0,0]
        cluster1 = [central, p1, p2, p3, p4]

        q1 = [-4, -4]
        q2 = [-6,-4]
        q3 = [-4,-6]
        q4 = [-6,-6]
        side = [-5,-5]
        cluster2 = [side, q1, q2, q3, q4]

        distant1 = [10, 10]
        distant2 = [11, 11]
        distant3 = [-11, -11]
        distant = [distant1, distant2, distant3]
        dataset = cluster1 + cluster2 + distant
        db = DBSCAN(dataset, euclidean, 4, 5)

        # should run this first so we get all epsilon neighbor for all dataset
        db.run_all_epsilon_neighbor()

        # is reachable false because the point is not the neighbor and
        # does not have minimum point
        questioned_point = db.dataset[-2]
        reference_point = db.dataset[-1]
        is_reachable = db.is_directly_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, False)

        # is reachable false because the point while contain its neighbor
        # does not have minimum point
        questioned_point = db.dataset[-2]
        reference_point = db.dataset[-3]
        is_reachable = db.is_directly_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, False)


        # is reaachable false because while it contain minimum point,
        # it does not belong to same neighbor
        questioned_point = db.dataset[-3]
        reference_point = db.dataset[0]
        is_reachable = db.is_directly_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, False)

        # is reachable is true because it is the neighbor,
        # it also has sufficient neighbor
        questioned_point = db.dataset[1]
        reference_point = db.dataset[0]
        is_reachable = db.is_directly_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, True)

    def test_is_density_reachable(self):
        db = DBSCAN([], euclidean, 1, 5)
        points = []

        # First case
        # no neighbor connected from reference point to questioned point
        is_reachable = db.is_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, False)

        # Second case
        # neighbor connected from rference point and is the part of 
        # directly density reachable
        is_reachable = db.is_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, True)

        # Third case
        # neighbor connected from reference point and is not the part of directly density reachable
        is_reachable = db.is_density_reachable(questioned_point, reference_point)
        self.assertEqual(is_reachable, True)
        
        

        

if __name__ == "__main__":
    unittest.main()

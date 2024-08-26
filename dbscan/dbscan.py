import unittest
import random
from point import Point
from typing import List, Tuple, Callable
from numbers import Number
import statistics

# an DBSCAN interface for sklearn like unsupervised learning


class DBSCAN:
    # in the original paper, min point and epsilon are automatically generated
    # based on dataset properties.
    def __init__(self, dataset, distance_func, min_point, epsilon):
        self.dataset = [Point(d) for d in dataset]
        self.epsilon = epsilon
        self.distance_func = distance_func
        self.min_point = min_point
        
    # definition 1
    def run_epsilon_neighbor(self, reference: Point):
        for point in self.dataset:
            if point == reference:
                continue
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
        return is_neighbor and has_min_point

    # definition 3
    def is_density_reachable(self, questioned_point: Point, reference_point: Point, already_visited: List[Point]) -> bool:
        # if the questioned point is directly  density reachbale then we dont need to search
        # connection via its neighbor
        if self.is_directly_density_reachable(questioned_point, reference_point):
            return True

        # this should be recusively find all possible neighbor. the exhaustive list would taken a long time.
        # we would optimise this function later.
        # there are also possibilities of densitiy reachable create a circular function
        # therefore create infinite recursion
        for neighbor in reference_point.get_neighbors():
            # neighbor should also fulfill the minimum point to proceed
            # remember, we need two requirement, neighbor epsilon and minimum point for reference point
            if not self.is_directly_density_reachable(neighbor, reference_point):
                continue
            if neighbor in already_visited:
                continue
            if self.is_directly_density_reachable(questioned_point, neighbor):
                return True
            already_visited.append(neighbor)
            if self.is_density_reachable(questioned_point, neighbor, already_visited):
                return True
        return False

    # definition 4
    def density_connected(self, questioned_point: Point, reference_point: Point) -> bool:
        if self.is_directly_density_reachable(questioned_point, reference_point):
            return True
        for neighbor in reference_point.get_neighbors():
            # in case you are wondering, yes it is reversed order from is_density_reachable
            # the edge in density connected should not have min point but point it connect
            # does need to be epsilon neighbor and min points
            if not self.is_directly_density_reachable(reference_point, neighbor):
                continue
            # this detect whether we reach core points, if we can reach core points
            # we can use the is_density_reachable to check whether we can
            # reach other edge or not
            ddreachable = self.is_directly_density_reachable(reference_point, neighbor)
            dreachable = self.is_density_reachable(questioned_point, neighbor, [])
            if ddreachable and dreachable:
                return True
        return False 

    def not_visited(self, already_visited: List[Point]) -> List[Point]:
        if len(already_visited) == 0:
            return self.dataset
        collect = []
        for point in self.dataset:
            if point not in already_visited:
                collect.append(point)
        return collect

    
    def random_pick_point(self, already_visited, determined = True) -> Point:
        # in the actual paper, picking point done by random choice
        # unf, picking random point would make the test hard and make pin pointing the bug harder
        # so we would use deterministic approach which can be toggled
        if determined:
            for pick in self.not_visited(already_visited):
                if pick.cluster is None and pick.is_noise is False:
                    return pick
            return None

        while True:
            n = self.not_visited(already_visited)
            rand_int = random.randint(0, len(n) - 1)
            # only pick point that not declared as a noise and
            # not belong to any cluster
            pick = n[rand_int]
            if pick.cluster is None and pick.is_noise is False:
                return pick

    def recursive_find(self, point: Point, cluster_count: int, already_visited: List[Point] = []) -> bool:
        # if a point does not have lower neighbor than assigned point, then it can be considered as noise
        # however, this also can be, potentially, considered as a border point so this can be 
        # override in other iteration; this would fulfill definition 6
        if point.neighbor_size() < self.min_point and point.cluster is None:
            point.is_noise = True
            already_visited.append(point)
            return False

        point.assign_cluster(cluster_count)
        for neighbor in point.get_neighbors():
            if neighbor in already_visited:
                continue
            # if the neighbor is density reachable, then we can assign the cluster
            # in definition 5, it is called maximality if it is density reachable
            # and it is called connectivity if it is density connected
            if self.is_density_reachable(neighbor, point, already_visited) or self.density_connected(neighbor, point):
                neighbor.assign_cluster(cluster_count)
                # when a point assigned to a cluster, previous attempt to mark it as a noise would be nulled
                neighbor.is_noise = False
            already_visited.append(neighbor)
            self.recursive_find(neighbor, cluster_count, already_visited)
        return True

    # definition 5 (cluster) and 6 (noise)
    def classification(self) -> List[int]:
        cluster_count = 1
        already_visited = []
        clusters = []
        while True:
            if len(already_visited) == len(self.dataset):
                break
            point = self.random_pick_point(already_visited)
            if self.recursive_find(point, cluster_count, already_visited):
                clusters.append(cluster_count)
                cluster_count += 1
        return clusters

    # get sorted K-distance based on reference point
    # this function will calculate all distance from a reference point to the rest of database
    # It will return array two value pair, first is the Kth order and its distance from reference point
    # we can do this by finding the distance to each point and reordering it
    def get_sorted_k_distance(self, reference: Point) -> List[Tuple[int, float]]:
        container = []
        for d in self.dataset:
            if d == reference:
                continue
            container.append([d.id, self.distance_func(reference, d)])
        return sorted(container, key=lambda c: c[1], reverse=True)

    # this is a numerical and approximation approach to an empirical data
    # we can reuse this several time if we want to get derivative of derivative
    def get_pseudo_derivative(self, array: List[Number]):
        container = []
        for index in range(1, len(array)):
            container.append(array[index] - array[index-1])
        return container

    # the cluster would get stablized i.e. does not have much changes in distance
    # in respect to reference point. therefore we can consider minimum point value
    # is the index of sorted k distance array when stablized.
    #
    # this function assume that array is already sorted. Would return
    # wrong index if not sorted. This function also assume that clusters
    # always have more member than noise
    def get_stabilized_index(self, array: List[Number]) -> int:
        if len(array) < 3:
            return 1
        avg = statistics.mean(array)
        std_dev = statistics.stdev(array)
        count = 0
        for index, value in enumerate(array):
            is_inside_range = value < avg + std_dev and value > avg - std_dev
            if is_inside_range:
                count += 1
            if not is_inside_range:
                count = 0
            if count > 3:
                return index
        return 0

    # work similiarly like get_stabilized_index but detecting non standard changes
    # return the index of first significant change
    #
    # this function assume that array is already sorted. Would return
    # wrong index if not sorted
    def get_elbow_index(self, array: List[Number]) -> int:
        if len(array) < 3:
            return 1
        avg = statistics.mean(array)
        std_dev = statistics.stdev(array)
        # unlike stabilized index, once we get the significant change we immediately
        # assign to previous index it
        for index, value in enumerate(array):
            if value > avg + std_dev or value < avg - std_dev:
                return index-1
        return 0

    # in the original paper, the authors decide that the min point is 4 based on their
    # observation, there are few ways we can improve this but thats for other time
    # however this is only tested in two dimensional data, other article
    # define that min_points should be equal to data dimension + 1
    # but give no further explanation why is it the way
    def generate_min_points(self, use_default=True):
        if use_default:
            return 4
        point = self.random_pick_point([], True)
        k_dist = self.get_sorted_k_distance(point)
        k_dist_derivative = self.get_pseudo_derivative([i[1] for i in k_dist])
        stabilized_index = self.get_stabilized_index(k_dist_derivative)
        return stabilized_index

    # in the original paper, the epsilon decided by context and user wisdom
    # In addition, it also mentioned that sorted k-dist can be helpful to determine
    # the optimal epsilon. Other literature and discussion points out that the "elbow"
    # of sorted k-dist ascending graph is the optimal value of epsilon since it is represent the first
    # significant distance between reference point and other cluster. This function
    # assume that we will pick a cluster point instead of noise.
    def generate_epsilon(self) -> float:
        point = self.random_pick_point([], True)
        k_dist = self.get_sorted_k_distance(point)
        rreversed = sorted(k_dist, key=lambda x: x[1])
        reversed_derivative = self.get_pseudo_derivative([i[1] for i in rreversed])
        elbow = self.get_elbow_index(reversed_derivative)
        return rreversed[elbow][1]


# an sklearn derivative of DBSCAN so it has same properties and methods so it changable with sklearn class
class DBSCANSklearn(DBSCAN):
    def __init__(self, eps=0.3, min_samples=10, distance='euclidean'):
        self.epsilon = eps
        self.min_point = min_samples
        self.labels_ = None
        self.distance = self.pick_distance(distance)

    def pick_distance(self, distance) -> Callable[[Point, Point], float]:
        def euclidean(p1, p2):
            return ((p1.data[0] - p2.data[0]) ** 2 + (p1.data[1] - p2.data[1]) ** 2 ) ** .5
        return euclidean

    def fit(self, X) -> bool:
        super().__init__(X, self.distance, self.min_point, self.epsilon)
        super().run_all_epsilon_neighbor()
        clusters = super().classification()
        self.labels_ = clusters
        return True

# example distance calculation, should be able to replace with other distance function
def euclidean(p1, p2):
    return ((p1.data[0] - p2.data[0]) ** 2 + (p1.data[1] - p2.data[1]) ** 2 ) ** .5


# contains all test unit for DBSCAN component from its base definition to actual clustering
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
        # actual db scan algorithm does not require this
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
        cluster1 = [
            [0, 0], # center cluster
            [0, 2], # edge 
            [2, 0],
            [-2, 0],
            [0, -2],
            [0, 6], # not epsilon neighbor of center cluster but reachable via edge
        ]

        cluster2 = [
                [-6,-6], # center cluster
                [-7,-6], # edge
                [-6,-7],
                [-5,-6],
                [-6,-5],
        ]

        noise = [
            [10, 10],
            [-10, -10],
        ]

        all_points = cluster1 + cluster2 + noise

        # it means it require at least four other point and other point should be at least
        # 5 distance point. Since we are using euclidean here, then it include all point
        # within radius of 2.5 from reference point.
        db = DBSCAN(all_points, euclidean, 4, 5)

        # should run this first so we get all epsilon neighbor for all dataset
        # actual db scan algorithm does not require this
        db.run_all_epsilon_neighbor()

        # First case
        # no neighbor connected from reference point to questioned point
        reference_point = db.dataset[0]
        questioned_point = db.dataset[-1]
        is_reachable = db.is_density_reachable(questioned_point, reference_point, [])
        self.assertEqual(is_reachable, False)

        # Second case
        # neighbor connected from rference point and is the part of 
        # directly density reachable
        reference_point = db.dataset[0]
        questioned_point = db.dataset[1]
        is_reachable = db.is_density_reachable(questioned_point, reference_point, [])
        self.assertEqual(is_reachable, True)

        # Third case
        # neighbor connected from reference point and is not the part of directly density reachable
        reference_point = db.dataset[0]
        questioned_point = db.dataset[5]
        is_reachable = db.is_density_reachable(questioned_point, reference_point, [])
        self.assertEqual(is_reachable, True)
        
        # Fourth case
        # reverse of the third case, should be false since it does not have symmetry
        reference_point = db.dataset[5]
        questioned_point = db.dataset[0]
        is_reachable = db.is_density_reachable(questioned_point, reference_point, [])
        self.assertEqual(is_reachable, False)

    def test_density_connected(self):
        cluster1 = [
            [0, 0], # center cluster
            [0, 2], # edge 
            [2, 0],
            [-2, 0],
            [0, -2],
            [0, 6], # not epsilon neighbor of center cluster but reachable via edge
            [-6, 0], # not epsilon neighbor of center cluster but reachable via edge
        ]

        distant = [
            [10,10]
        ]

        all_points = cluster1 + distant
        db = DBSCAN(all_points, euclidean, 4, 5)

        # should run this first so we get all epsilon neighbor for all dataset
        # actual db scan algorithm does not require this
        db.run_all_epsilon_neighbor()

        # a directly desnsity neighbor should also be density connected
        questioned_point = db.dataset[0]
        reference_point = db.dataset[1]
        is_connected = db.density_connected(questioned_point, reference_point)
        self.assertEqual(is_connected, True)

        # both should be connected via center cluster while both are not
        # epsilon neighbor of center cluster
        questioned_point = db.dataset[-2]
        reference_point = db.dataset[-3]
        is_connected = db.density_connected(questioned_point, reference_point)
        self.assertEqual(is_connected, True)

        # a distant point should not be considered as density connected
        # epsilon neighbor of center cluster
        questioned_point = db.dataset[-1]
        reference_point = db.dataset[-2]
        is_connected = db.density_connected(questioned_point, reference_point)
        self.assertEqual(is_connected, False)

    def test_classification(self):
        # all of this should belong to the same cluster
        cluster1 = [
            [0, 0], # center cluster
            [0, 2], # edge 
            [2, 0],
            [-2, 0],
            [0, -2],
            [0, 6], # not epsilon neighbor of center cluster but reachable via edge
            [-6, 0], # not epsilon neighbor of center cluster but reachable via edge
        ]

        # all of this should belong to the same cluster
        cluster2 = [
                [-16,-16], # center cluster
                [-17,-16], 
                [-16,-17],
                [-15,-16],
                [-16,-15],
        ]

        # should not be assigned to any cluster
        noise = [
            [200,200],
            [-200,-200],
        ]

        all_points = cluster1 + cluster2 + noise

        db = DBSCAN(all_points, euclidean, 4, 5)
        db.run_all_epsilon_neighbor()

        clusters = db.classification()
        self.assertListEqual(clusters, [1, 2])
        
        # cluster might have different order depending on random pick
        # all we need is to ensure whether a group belong to same cluster or not
        # cluster not tested since it is our reference
        first_cluster = db.dataset[0].cluster
        second_cluster = db.dataset[7].cluster
        self.assertNotEqual(first_cluster, second_cluster)

        self.assertEqual(db.dataset[1].cluster, first_cluster)
        self.assertEqual(db.dataset[2].cluster, first_cluster)
        self.assertEqual(db.dataset[3].cluster, first_cluster)
        self.assertEqual(db.dataset[4].cluster, first_cluster)
        self.assertEqual(db.dataset[5].cluster, first_cluster)
        self.assertEqual(db.dataset[6].cluster, first_cluster)

        self.assertEqual(db.dataset[8].cluster, second_cluster)
        self.assertEqual(db.dataset[9].cluster, second_cluster)
        self.assertEqual(db.dataset[10].cluster, second_cluster)
        self.assertEqual(db.dataset[11].cluster, second_cluster)

        self.assertEqual(db.dataset[12].cluster, None)
        self.assertEqual(db.dataset[12].is_noise, True)
        self.assertEqual(db.dataset[13].cluster, None)
        self.assertEqual(db.dataset[13].is_noise, True)

    def test_get_sorted_k_distance(self):
        # all of this should belong to the same cluster
        cluster1 = [
            [0, 0], # center cluster
            [0, 2.9], # edge 
            [2.7, 0],
            [-2.4, 0],
            [0, -2.2]
        ]
        db = DBSCAN(cluster1, euclidean, 4, 5)
        sortedd = db.get_sorted_k_distance(db.dataset[0])

        self.assertEqual(len(sortedd), len(cluster1)-1)
        
        # the first one should be the most further
        self.assertEqual(sortedd[0][0], db.dataset[1].id)
        self.assertEqual(sortedd[1][0], db.dataset[2].id)
        self.assertEqual(sortedd[2][0], db.dataset[3].id)
        self.assertEqual(sortedd[3][0], db.dataset[4].id)

    def test_generate_min_points(self):
        # all of this should belong to the same cluster
        cluster1 = [
            [0, 0], # center cluster
            [0, 2], # edge 
            [2, 0],
            [-2, 0],
            [0, -2],
            [0, 6], # not epsilon neighbor of center cluster but reachable via edge
            [-6, 0], # not epsilon neighbor of center cluster but reachable via edge
        ]

        # all of this should belong to the same cluster
        cluster2 = [
            [-16,-16], # center cluster
            [-17,-16], 
            [-16,-17],
            [-15,-16],
            [-16,-15],
        ]

        # should not be assigned to any cluster
        noise = [
            [200,200],
            [-200,-200],
            [-400,-700],
            [-600,-800],
            [-650,-800],
        ]
        all_points = cluster1 + cluster2 + noise
        db = DBSCAN(all_points, euclidean, 4, 5)
        index = db.generate_min_points(False)
        self.assertEqual(index, len(noise) + 3)

    def test_generate_epsilon(self):
        # all of this should belong to the same cluster
        cluster1 = [
            [0, 0], # center cluster
            [0, 2], # edge 
            [2, 0],
            [0, -2],
            [-2, 0], 
        ]

        # all of this should belong to the same cluster
        cluster2 = [
                [-16,-16], # center cluster
                [-17,-16], 
                [-16,-17],
                [-15,-16],
                [-16,-15],
        ]

        # should not be assigned to any cluster
        noise = [
            [30,2],
            [-2,-30],
        ]
        all_points = cluster1 + cluster2 + noise
        db = DBSCAN(all_points, euclidean, 4, 5)
        distance = db.generate_epsilon()
        self.assertLessEqual(distance, 2)

    def test_dbscan_sklearn(self):
        # use the same dataset for clusters since it is essentially called classification from parent class
        # all of this should belong to the same cluster
        cluster1 = [
            [0, 0], # center cluster
            [0, 2], # edge 
            [2, 0],
            [-2, 0],
            [0, -2],
            [0, 6], # not epsilon neighbor of center cluster but reachable via edge
            [-6, 0], # not epsilon neighbor of center cluster but reachable via edge
        ]

        # all of this should belong to the same cluster
        cluster2 = [
                [-16,-16], # center cluster
                [-17,-16], 
                [-16,-17],
                [-15,-16],
                [-16,-15],
        ]

        # should not be assigned to any cluster
        noise = [
            [200,200],
            [-200,-200],
        ]

        all_points = cluster1 + cluster2 + noise
        skl =  DBSCANSklearn(5, 4, 'euclidean')
        skl.fit(all_points)
        self.assertEqual(len(skl.labels_), 2)

        
if __name__ == "__main__":
    unittest.main()

from __future__ import annotations
from typing import List
from counter import Counter
import unittest

class Point:
    def __init__(self, data):
        # always call counter when init, since it is a singleton
        # it would always be created only once, performance impact is minimal
        self.id = Counter().increment()
        self.data = data
        self.neighbors = []
        return

    def __eq__(self, other: Point) -> bool:
        return self.id == other.id

    def add_neighbor(self, data_point: Point):
        self.neighbors.append(data_point)

    def has_neighbor(self, data_point: Point) -> bool:
        return data_point in self.neighbors

    def neighbor_size(self) -> int:
        return len(self.neighbors)
    
    def get_neighbors(self) -> List[Point]:
        return self.neighbors

class TestPoint(unittest.TestCase):
    def test_adding_neighbors(self):
        first_point = Point([1,2])
        second_point = Point([2,3])
        first_point.add_neighbor(second_point)
        self.assertEqual(first_point.neighbor_size(), 1)

    def test_check_equality(self):
        arr = [1,2]
        first_point = Point(arr)
        second_point = Point(arr)
        self.assertNotEqual(first_point, second_point)
        
        another_first_point = first_point
        self.assertEqual(first_point, another_first_point)

    def test_has_neighbors(self):
        first_point = Point([1,2])
        second_point = Point([2,3])
        third_point = Point([5,3])
        first_point.add_neighbor(second_point)
        self.assertEqual(first_point.has_neighbor(second_point), True)
        self.assertEqual(first_point.has_neighbor(third_point), False)

if __name__ == "__main__":
    unittest.main()

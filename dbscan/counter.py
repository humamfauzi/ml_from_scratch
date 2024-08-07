import unittest

class Counter:
    _instance = None
    _count = 0
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Counter, cls).__new__(cls)
        return cls._instance

    def increment(self):
        self._count += 1
        return self.get_count()

    def get_count(self):
        return self._count

    def reset(self):
        self._count = 0

class TestCounter(unittest.TestCase):
    def test_instantiation(self):
        ct = Counter()
        self.assertEqual(ct.get_count(), 0)

    def test_not_reset_in_different_instantiation(self):
        ct = Counter()
        self.assertEqual(ct.increment(), 1)

        ct2 = Counter()
        self.assertEqual(ct.increment(), 2)

    def test_reset(self):
        ct = Counter()
        ct.reset()
        self.assertEqual(ct.get_count(), 0)

if __name__ == "__main__":
    unittest.main()

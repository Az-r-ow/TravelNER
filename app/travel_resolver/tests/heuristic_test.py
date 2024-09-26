import unittest
from travel_resolver.libs.pathfinder.heuristic import euclidean_distance,haversine_distance

class TestHeuristic(unittest.TestCase):

    def setUp(self):
        self.latParis, self.lonParis = 48.8566, 2.3522
        self.latLyon, self.lonLyon = 45.7640, 4.8357

    def test_euclidian_distance(self):
        expected_distance = 0.06922589910147367
        distance = euclidean_distance(self.latParis, self.lonParis, self.latLyon, self.lonLyon)
        self.assertEqual(distance, expected_distance)

    def test_haversin_distance(self):
        expected_distance = 391.4989316742569
        distance = haversine_distance(self.latParis, self.lonParis, self.latLyon, self.lonLyon)
        self.assertEqual(distance, expected_distance)

if __name__ == '__main__':
    unittest.main()

import unittest
from travel_resolver.libs.pathfinder.heuristic import haversine_distance,get_minutes_from_distance,euclidean_distance

class TestHeuristic(unittest.TestCase):

    def setUp(self):
        self.latParis, self.lonParis = 48.8566, 2.3522
        self.latLyon, self.lonLyon = 45.7640, 4.8357

    def test_euclidean_distance(self):
        expected_distance = 441.0382031754888
        distance = euclidean_distance(self.latParis, self.lonParis, self.latLyon, self.lonLyon)
        self.assertEqual(distance, expected_distance)

    def test_haversine_distance(self):
        expected_distance = 391.4989316742569
        distance = haversine_distance(self.latParis, self.lonParis, self.latLyon, self.lonLyon)
        self.assertEqual(distance, expected_distance)

    def test_get_minutes_from_distance(self):
        expected_time_euclidean = 264.62292190529325
        euclidean_dist = euclidean_distance(self.latParis, self.lonParis, self.latLyon, self.lonLyon)
        time_from_euclidean_distance = get_minutes_from_distance(euclidean_dist,True)

        expected_time_haversine = 234.89935900455413
        haversine_dist = haversine_distance(self.latParis, self.lonParis, self.latLyon, self.lonLyon)
        time_from_haversine_distance = get_minutes_from_distance(haversine_dist,True)

        self.assertEqual(time_from_euclidean_distance, expected_time_euclidean)
        self.assertEqual(time_from_haversine_distance, expected_time_haversine)

if __name__ == '__main__':
    unittest.main()

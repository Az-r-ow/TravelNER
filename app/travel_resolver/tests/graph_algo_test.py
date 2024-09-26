import unittest
from travel_resolver.libs.pathfinder.graph import Graph
from typing import Dict

class TestGraphAlgo(unittest.TestCase):

    def setUp(self):
        # Initialisation d'un graphe d'exemple et des heuristiques pour les tests
        self.graph_data: Dict[str, Dict[str, float]] = {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }
        self.heuristic_data: Dict[str, float] = {
            'A': 7,
            'B': 5,
            'C': 2,
            'D': 0
        }
        self.graph = Graph(self.graph_data)

    def test_RunDijkstra(self):
        # Test du chemin le plus court avec Dijkstra depuis le noeud 'A'
        expected_distances = {'A': 0, 'B': 1, 'C': 3, 'D': 4}
        distances = self.graph.RunDijkstra('A')
        self.assertEqual(distances, expected_distances)

    def test_RunAStar(self):
        # Test de l'algorithme A* depuis le noeud 'A' vers 'D'
        expected_path = ['A', 'B', 'C', 'D']
        expected_costs = {'A': 0, 'B': 1, 'C': 3, 'D': 4}

        path, costs = self.graph.RunAStar('A', 'D', self.heuristic_data)
        self.assertEqual(path, expected_path)
        self.assertEqual(costs, expected_costs)

if __name__ == '__main__':
    unittest.main()

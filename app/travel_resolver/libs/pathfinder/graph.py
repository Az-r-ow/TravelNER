import heapq
from typing import Dict, List, Tuple

Graph = Dict[str, Dict[str, float]]
Heuristic = Dict[str, float]

class Graph():
    def __init__(self, graph_data: Graph):
        """
        Initialize base private attributes with sample values.
        """
        self.graph: Graph = graph_data

    def RunDijkstra(self, start: str) -> Dict[str, int]:
        """
        Run Dijkstra algorithm to get all shortest ways to every node from start node.
        Args:
            start (str): Name of the start node.

        Returns:
            (Dict[str, int]): The length of the shortest way to every node of the graph from start node.
        """
        # Initialisation des distances avec l'infini
        distances: Dict[str, float] = {node: float('inf') for node in self.graph}
        distances[start] = 0  # Distance au point de départ est 0
        priority_queue: List[Tuple[float, str]] = [(0, start)]  # (distance, noeud)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # Si la distance actuelle est plus grande que la distance déjà trouvée, on continue
            if current_distance > distances[current_node]:
                continue

            # Vérifier les voisins du noeud courant
            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight

                # Si une distance plus courte est trouvée
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

    def RunDijkstraBetweenTwoNodes(self, start: str, end: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Run Dijkstra's algorithm to get the shortest path from start to end node, including the total weight.
        
        Args:
            start (str): Name of the start node.
            end (str): Name of the end node.
            
        Returns:
            (Tuple[List[str], Dict[str, float]]): A tuple where the first element is the path to the goal and
            the second element is a dictionary with the cost to reach each node.
        """
        # Initialisation des distances avec l'infini
        distances: Dict[str, float] = {node: float('inf') for node in self.graph}
        distances[start] = 0  # Distance au point de départ est 0

        # Prédecesseurs pour reconstruire le chemin
        predecessors: Dict[str, str] = {node: None for node in self.graph}

        # File de priorité pour traiter les noeuds
        priority_queue: List[Tuple[float, str]] = [(0, start)]  # (distance, noeud)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # Si on arrive au noeud final, on peut arrêter plus tôt
            if current_node == end:
                break

            # Si la distance actuelle est plus grande que celle déjà trouvée, on continue
            if current_distance > distances[current_node]:
                continue

            # Vérifier les voisins du noeud courant
            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight

                # Si une distance plus courte est trouvée
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node  # On enregistre le prédecesseur
                    heapq.heappush(priority_queue, (distance, neighbor))

        # Reconstruction du chemin de 'start' à 'end'
        path = []
        current_node = end
        while current_node is not None:
            path.insert(0, current_node)
            current_node = predecessors[current_node]

        # Si le chemin est vide ou n'atteint pas le noeud de fin, renvoyer None
        if distances[end] == float('inf'):
            return float('inf'), []

        return  path,distances

    def RunAStar(self, start: str, goal: str, heuristic: Heuristic) -> Tuple[List[str], Dict[str, float]]:
        """
        Run A* algorithm to find the shortest path and the costs to each step.
        Args:
            start (str): Name of the start node.
            goal (str): Name of the goal node.
            heuristic (Heuristic): Estimation of distance from every node to goal node.

        Returns:
            (Tuple[List[str], Dict[str, float]]): A tuple where the first element is the path to the goal and
            the second element is a dictionary with the cost to reach each node.
        """
        # Initialisation des distances avec l'infini
        g_score: Dict[str, float] = {node: float('inf') for node in self.graph}
        g_score[start] = 0

        # f_score = g_score + h (heuristique)
        f_score: Dict[str, float] = {node: float('inf') for node in self.graph}
        f_score[start] = heuristic[start]

        priority_queue: List[Tuple[float, str]] = [(f_score[start], start)]  # (f_score, noeud)
        came_from: Dict[str, str] = {}  # Pour reconstruire le chemin

        while priority_queue:
            _, current = heapq.heappop(priority_queue)

            # Si on atteint l'objectif, on peut reconstruire le chemin
            if current == goal:
                return self.__reconstructPath(came_from, current), g_score

            # Vérifier les voisins du noeud courant
            for neighbor, weight in self.graph[current].items():
                tentative_g_score = g_score[current] + weight

                if tentative_g_score < g_score[neighbor]:
                    # Un chemin plus court vers ce voisin a été trouvé
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                    heapq.heappush(priority_queue, (f_score[neighbor], neighbor))

        return [], g_score  # Si aucun chemin n'a été trouvé, renvoyer un chemin vide et les coûts

    def __reconstructPath(self, came_from: Dict[str, str], current: str) -> List[str]:
        """
        Private function only used in A* algorithm
        Reconstruct the path from the start node to the current node.
        Args:
            came_from (Dict[str, str]): Dictionary tracking the previous node for each node.
            current (str): Current node.

        Returns:
            (List[str]): The path from the start node to the current node.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]  # Inverser pour avoir le chemin dans le bon ordre

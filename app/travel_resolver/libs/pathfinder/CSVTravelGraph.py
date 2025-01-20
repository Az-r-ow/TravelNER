import pandas as pd
from typing import Dict

class CSVTravelGraph():
    def __init__(self, csv_file: str):
        """
        Read csv and create the graph in function of the given mode
        """
        self.csv = pd.read_csv(csv_file, sep="\t")
        self.data = self.generateGraph()
        
    def generateGraph(self):
        """
        Create a graph by browsing the data retrieved in the csv
        Returns:
            (Dict[str, Dict[str, float]]): The graph
        """
        graph: Dict[str, Dict[str, float]] = {}
        
        def addTravelToTheGraph(depart, arrive, duree):
            if depart in graph:
                graph[depart][arrive] = duree
            else:
                graph[depart] = {arrive: duree}
            
        for index, row in self.csv.iterrows():
            trip_id, trajet, duree = row
            points = trajet.split(' - ')
            depart = ' - '.join(points[:-1])
            arrive = points[-1]
            duree = float(duree)
            
            addTravelToTheGraph(depart, arrive, duree)
            addTravelToTheGraph(arrive, depart, duree)
            
        return graph
        
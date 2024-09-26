import math

def euclidean_distance(lat1 : float, lon1 : float, lat2 : float, lon2 : float):
    """
    Calculate the Euclidean distance between two points given by their latitudes and longitudes.
    The coordinates are in degrees.
    Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the first point.

        Returns:
            (float): The Euclidean distance (radians) between the two points.
    """
    # Conversion des degrés en radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    # Différence des coordonnées
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Distance euclidienne (approximation plane)
    distance = math.sqrt(delta_lat**2 + delta_lon**2)
    
    return distance

def haversine_distance(lat1 : float, lon1 : float, lat2 : float, lon2 : float):
    """
    Calculates the distance as the crow flies between two points on the Earth
    (coordinates specified in degrees) using the Haversine formula.
    Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the first point.

        Returns:
            (float): The distance as the crow flies (kilometers) between the two points.
    """
    # Rayon de la Terre en kilomètres
    R = 6371.0
    
    # Conversion des degrés en radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    # Différences des coordonnées
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Formule de haversine
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance à vol d'oiseau
    distance = R * c
    
    return distance

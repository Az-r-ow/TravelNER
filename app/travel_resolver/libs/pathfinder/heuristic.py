import math
import travel_resolver.libs.pathfinder.variables as var

def euclidean_distance(lat1 : float, lon1 : float, lat2 : float, lon2 : float) -> float:
    """
    Calculate the Euclidean distance between two points given by their latitudes and longitudes.
    The coordinates are in degrees.
    Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the first point.

        Returns:
            (float): The Euclidean distance (kilometers) between the two points.
    """
    # Conversion des degrés en radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    # Différence des coordonnées
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Distance euclidienne (approximation plane)
    distance = math.sqrt(delta_lat**2 + delta_lon**2)
    
    return distance*var.EARTH_RADIUS

def haversine_distance(lat1 : float, lon1 : float, lat2 : float, lon2 : float) -> float:
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
    distance = var.EARTH_RADIUS * c
    
    return distance

def get_minutes_from_distance(distance : float, needConversion : bool) -> float:
    """
    Get minutes estimation from heuristic distance computed by in kilometers or radians
    Args:
            distance (float): The converted distance.
            needConversion (bool): Determine if conversion is needed.
                If true, this distance is in kilometers.
                Else, it is in radians.

        Returns:
            (float): The approximated minutes needed to travel the given distance.
    """
    speed : float = var.AVERAGE_KM_H_TRAIN_SPEED if needConversion else var.AVERAGE_RAD_MIN_TRAIN_SPEED
    conversionRatio : float = 60 if needConversion else 1
    return (distance/speed)*conversionRatio
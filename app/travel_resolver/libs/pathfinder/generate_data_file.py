import csv

BASE_PATH = "../data/sncf/"

# Charger les informations des gares depuis le fichier "sncf_stations_databases"
def charger_infos_gares(fichier_stations):
    infos_gares = {}
    with open(fichier_stations, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nom_gare = row['LIBELLE'].strip()
            infos_gares[nom_gare] = {
                'commune': row['COMMUNE'].strip(),
                'latitude': row['Y_WGS84'].strip(),
                'longitude': row['X_WGS84'].strip()
            }
    return infos_gares

# Fonction pour trouver une gare dans les infos_gares avec une correspondance partielle
def trouver_gare_par_nom(nom_gare, infos_gares):
    for libelle in infos_gares:
        if nom_gare in libelle:
            return infos_gares[libelle]
    return None

# Lire le fichier "timetables.csv" et générer le nouveau fichier avec les informations de gares
def creer_nouveau_fichier(fichier_timetables, fichier_stations, fichier_sortie):
    # Charger les informations des gares
    infos_gares = charger_infos_gares(fichier_stations)

    # Créer un ensemble pour stocker les gares déjà inscrites
    gares_deja_inscrites = set()
    gares_sans_info = set()

    # Ouvrir le fichier de sortie
    with open(fichier_sortie, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Écrire les en-têtes
        writer.writerow(['Nom de la gare', 'Commune', 'Latitude', 'Longitude'])

        # Lire le fichier des horaires ("timetables.csv")
        with open(fichier_timetables, newline='', encoding='utf-8') as timetablefile:
            reader = csv.DictReader(timetablefile, delimiter='\t')
            for row in reader:
                # Extraire les gares de départ et d'arrivée
                gares = row['trajet'].split(' - ')
                gare_depart = gares[0].strip()
                gare_arrivee = gares[1].strip()

                # Chercher les infos pour chaque gare dans le fichier des stations
                for gare in [gare_depart, gare_arrivee]:
                    nom_reduit = gare.replace("Gare de ", "").strip()
                    
                    # Essayer de trouver les infos avec correspondance exacte ou partielle
                    info = trouver_gare_par_nom(nom_reduit, infos_gares)
                    
                    if info and gare not in gares_deja_inscrites:
                        writer.writerow([gare, info['commune'], info['latitude'], info['longitude']])
                        # Ajouter la gare à l'ensemble des gares déjà inscrites
                        gares_deja_inscrites.add(gare)
                    elif not info:
                        gares_sans_info.add(gare)
    for gare in gares_sans_info:
        print(f"Infos non trouvées pour la gare : {gare}")
    print("Nombre de gares sans informations : ",len(gares_sans_info))


# Exemple d'utilisation
fichier_timetables = BASE_PATH + 'timetables.csv'
fichier_stations = BASE_PATH + 'sncf_stations_database.csv'
fichier_sortie = BASE_PATH + 'gares_info.csv'

creer_nouveau_fichier(fichier_timetables, fichier_stations, fichier_sortie)

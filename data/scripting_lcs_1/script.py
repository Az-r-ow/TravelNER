import os
import sys
import csv
import random
from typing import List


def make_unique_lignes(f_in: str, f_out: str) -> int:
    """
    Delete all duplicate lignes of a file.

    Args:
        f_in (str): File path to analyse, must contain extension.
        f_out (str): File path containing result, must contain extension.

    Returns:
        (int): The number of duplicate lignes found.
    """

    seen_lignes: set = set()
    duplicates: int = 0

    with open(f_in, "r") as in_f, open(f_out, "w") as out_f:
        for ligne in in_f:
            if ligne not in seen_lignes:
                out_f.write(ligne)
                seen_lignes.add(ligne)
            else:
                duplicates += 1

    return duplicates


def count_file_lignes(f_path: str) -> int:
    """
    Count the number of lines in a file.

    Args:
        f_path (str): File path to analyse, must contain extension.

    Returns:
        (int): The number of lignes found.
    """

    with open(f_path, "r") as f:
        lignes = f.readlines()
        return len(lignes)


def get_cities() -> List:
    """
    Returns all cities from sncf db_file.

    Returns:
        (List): All cities present in file.
    """
    villes = []
    with open("../sncf_stations_database.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            villes.append(row["COMMUNE"])
    return villes


def generate_data(cities: List, file_out: str):
    """
    Generate dataset from template file.

    Args:
        cities (List): Cities from wich combinaison will generate.
        file_out (str): Output file, must contain extension.
    """

    used_comp = set()
    cities = get_cities()

    with open("data_unique_tmp.txt", "r") as f_template:
        template_ligne = f_template.readlines()

    with open(file_out, "w") as f_sortie:
        while len(used_comp) < 75000:

            arrival_city = random.choice(cities)
            departure_city = random.choice(cities)

            while arrival_city == departure_city:
                arrival_city = random.choice(cities)

            combinaison = (arrival_city, departure_city)

            if combinaison not in used_comp:
                used_comp.add(combinaison)
                for ligne in template_ligne:
                    new_ligne = ligne.replace("{depart}", departure_city)
                    new_ligne = new_ligne.replace("{arrivee}", arrival_city)
                    f_sortie.write(new_ligne)


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_in> <file_out>")
        sys.exit(1)
    else:
        file_in = sys.argv[1]
        file_out = sys.argv[2]

        duplicates: int = make_unique_lignes(file_in, "data_unique_tmp.txt")

        cities: List = get_cities()
        generate_data(cities, file_out)

        initial_ligne_number: int = count_file_lignes(file_in)
        unique_sentences_number: int = count_file_lignes("data_unique_tmp.txt")
        final_data_number: int = count_file_lignes(file_out)

        os.remove("data_unique_tmp.txt")

        print("Treatment is finished : ")
        print("     - Input file : " + file_in)
        print("     - Unitial number of lignes : " + str(initial_ligne_number))
        print("     - Number of duplicates found : " + str(duplicates))
        print("     - Unique sentence forms : " + str(unique_sentences_number))
        print("     - Output file : " + file_out)
        print("     - Final dataset size : " + str(final_data_number))


if __name__ == "__main__":
    main()

import os
import sys
import csv
import random
from typing import List


def make_unique_lines(f_in: str, f_out: str) -> int:
    """
    Delete all duplicate lines of a file.

    Args:
        f_in (str): File path to analyse, must contain extension.
        f_out (str): File path containing result, must contain extension.

    Returns:
        (int): The number of duplicate lines found.
    """

    seen_lines: set = set()
    duplicates: int = 0

    with open(f_in, "r") as in_f, open(f_out, "w") as out_f:
        for line in in_f:
            if line not in seen_lines:
                out_f.write(line)
                seen_lines.add(line)
            else:
                duplicates += 1

    return duplicates


def count_file_lines(f_path: str) -> int:
    """
    Count the number of lines in a file.

    Args:
        f_path (str): File path to analyse, must contain extension.

    Returns:
        (int): The number of lines found.
    """

    with open(f_path, "r") as f:
        lines = f.readlines()
        return len(lines)


def get_cities() -> List:
    """
    Returns all cities from sncf db_file.

    Returns:
        (List): All cities present in file.
    """
    cities = []
    with open("../sncf_stations_database.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            cities.append(row["COMMUNE"])
    return cities


def generate_data(cities: List, file_out: str, nb_samples: int):
    """
    Generate dataset from template file.

    Args:
        cities (List): Cities from wich combinaison will generate.
        file_out (str): Output file, must contain extension.
    """

    user_comb = set()
    cities = get_cities()
    print(len(cities))
    line_count = 0

    with open("data_unique_tmp.txt", "r") as f_template:
        template_line = f_template.readlines()

    with open(file_out, "w") as f_sortie:
        while line_count <= nb_samples:
            arrival_city = random.choice(cities)
            departure_city = random.choice(cities)

            while arrival_city == departure_city:
                arrival_city = random.choice(cities)

            combination = (arrival_city, departure_city)

            if combination not in user_comb:
                user_comb.add(combination)
                line = random.choice(template_line)
                new_line = line.replace("{depart}", departure_city)
                new_line = new_line.replace("{arrivee}", arrival_city)
                try:
                    n_chars_written = f_sortie.write(new_line)
                    if n_chars_written != len(new_line):
                        raise Exception("Error while writing line")
                    line_count += 1
                except Exception as e:
                    print(e)
                    print(new_line)


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_in> <file_out> <nb_sample>")
        sys.exit(1)
    else:
        file_in = sys.argv[1]
        file_out = sys.argv[2]
        nb_samples = int(sys.argv[3])

        duplicates: int = make_unique_lines(file_in, "data_unique_tmp.txt")

        cities: List = get_cities()

        generate_data(cities, file_out, nb_samples)

        initial_line_number: int = count_file_lines(file_in)
        unique_sentences_number: int = count_file_lines("data_unique_tmp.txt")
        final_data_number: int = count_file_lines(file_out)

        os.remove("data_unique_tmp.txt")

        print("Treatment is finished : ")
        print("     - Input file : " + file_in)
        print("     - Unitial number of lines : " + str(initial_line_number))
        print("     - Number of duplicates found : " + str(duplicates))
        print("     - Unique sentence forms : " + str(unique_sentences_number))
        print("     - Output file : " + file_out)
        print("     - Final dataset size : " + str(final_data_number))


if __name__ == "__main__":
    main()

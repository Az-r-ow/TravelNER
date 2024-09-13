from collections import Counter
import csv
from typing import List
import travel_resolver.libs.nlp.langage_detection.variables as var


def extract_data_from_csv(f_in: str, f_out: str):
    """
        Take a csv file containing strings and convert it
        into a csv file containig letter frequencies infos.

        Args:
            f_in (str): File path to analyse, must contain extension.
            f_out (str): File path containing result, must contain extension.
    """

    with open(f_in, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        with open(f_out, 'w', newline='') as output_csv:
            csv_writer = csv.writer(output_csv)

            for row in csv_reader:
                str = "".join(row).lower()
                modified_row = extract_data_from_string(str)
                csv_writer.writerow(modified_row)


def extract_data_from_string(str_in: str) -> List:
    """
        Retreive tab containing letter frequency informations
        and special char frequency of a given string.

        Args:
            str_in (str): String to analyse.

        Returns:
            (List): Tab containing special char and alphabetical frequencies.
    """
    str_data = []
    str_data = str_data + frequence_letters(str_in)
    str_data = str_data + frequence_char_part(str_in)
    return str_data


def frequence_letters(str_in: str) -> List:
    """
        Retreive tab containing letter frequency informations
        of a given string.

        Args:
            str_in (str): String to analyse.

        Returns:
            (List): Tab containing alphabetical char frequencies.
    """
    counter = Counter(str_in.lower())
    freq_tab = [round(counter.get(chr(i), 0) / len(counter) * 100, 2)
                for i in range(97, 123)]
    return freq_tab


def frequence_char_part(str_in: str) -> List:
    """
        Retreive tab containing special char frequency
        informations of a given string.

        Args:
            str_in (str): String to analyse.

        Returns:
            (List): Tab containing special char char frequencies.
    """

    counter = Counter(str_in.lower())
    freq_tab = [round(counter.get(char, 0) / len(str_in) * 100, 2)
                for char in var.SPECIAL_CHARS]
    return freq_tab


def main():
    for lang in var.TRAD_TARGETS:
        input_file = '../../assets/data/prompts/csv/'+lang+'_prompts.csv'
        output_csv_file = '../../assets/data/trainset/'+lang+'_trainset.csv'
        extract_data_from_csv(input_file, output_csv_file)


if __name__ == "__main__":
    main()

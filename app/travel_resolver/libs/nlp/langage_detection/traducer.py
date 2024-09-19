import csv
import deepl
import os
import travel_resolver.libs.nlp.langage_detection.variables as var


def traduce_into_csv(f_in: str, f_out: str, target_lang: str):
    """
    Take an input file that contains french text
    and translate it into a csv file.

    Args:
        f_in (str): File path to analyse, must contain extension.
        f_out (str): File path containing result, must contain extension.
        target_lang (str): Key representing output langage.
    """

    translator = deepl.Translator(os.getenv(var.ENV_AUTH_KEY))

    with open(f_in, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        with open(f_out, "w", newline="") as output_csv:
            csv_writer = csv.writer(output_csv)
            for row in csv_reader:
                str = "".join(row).lower()

                str = translator.translate_text(
                    str, target_lang=target_lang, source_lang=var.FR
                )
                modified_row = [str]
                csv_writer.writerow(modified_row)


def main():
    for lang in var.TRAD_TARGETS:
        source = "../../../../data/langage_detection/prompts/FR_prompts.csv"
        output_csv_file = "../../../../data/langage_detection/"
        output_csv_file += lang + "_prompts.csv"

        traduce_into_csv(source, output_csv_file, lang)


if __name__ == "__main__":
    main()

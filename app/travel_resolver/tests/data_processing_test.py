import unittest
from pathlib import Path
from travel_resolver.libs.nlp.data_processing import (
    get_tagged_content,
    convert_tagged_sentence_to_bio,
    from_tagged_file_to_bio_file,
    from_bio_file_to_examples,
)

bio_format_sentence = """Je O
voudrais O
voyager O
de O
Nice B-LOC-DEP
à O
Clermont B-LOC-ARR
Ferrand I-LOC-ARR
. O"""


class TestDataProcessing(unittest.TestCase):
    script_dir = Path(__file__).parent
    test_samples_dir = f"{script_dir}/test_samples"

    def test_get_tagged_content(self):
        simple_city = "Paris"
        input_simple = f"I am going to <Dep>{simple_city}<Dep>"
        result_simple = get_tagged_content(input_simple, "<Dep>")
        self.assertEqual(result_simple, simple_city)

        composed_city = "Los angeles"
        input_complex = f"I am going to <Dep>{composed_city}<Dep> and <Arr>London<Arr> <Dep>Paris<Dep>"
        result_complex = get_tagged_content(input_complex, "<Dep>")

        self.assertEqual(result_complex, composed_city)

        input_empty = "I am going to Paris"
        result_empty = get_tagged_content(input_empty, "<Dep>")

        self.assertEqual(result_empty, None)

        with self.assertRaises(ValueError):
            get_tagged_content(input_simple, "")

        with self.assertRaises(ValueError):
            get_tagged_content(input_simple, 123)

    def test_convert_tagged_sentence_to_bio(self):
        sentence = "Je voudrais voyager de <Dep>Nice<Dep> à <Arr>Clermont Ferrand<Arr>."
        tag_entities_pairs = [("<Dep>", "LOC-DEP"), ("<Arr>", "LOC-ARR")]

        result = convert_tagged_sentence_to_bio(sentence, tag_entities_pairs)

        self.assertEqual(result, bio_format_sentence)

    def test_from_tagged_file_to_bio(self):
        simple_input_file = f"{self.test_samples_dir}/simple_tagged_sentence.txt"
        expected_file = f"{self.test_samples_dir}/simple_tagged_sentence.bio"
        output_file = f"{self.test_samples_dir}/output.bio"
        tag_entities_pairs = [("<Dep>", "LOC-DEP"), ("<Arr>", "LOC-ARR")]

        from_tagged_file_to_bio_file(simple_input_file, output_file, tag_entities_pairs)

        with open(output_file, "r") as f:
            result = f.read()

        with open(expected_file, "r") as f:
            expected = f.read()

        self.assertEqual(result, expected)

        multiple_sentences__file = (
            f"{self.test_samples_dir}/multiple_tagged_sentences.txt"
        )
        expected_file = f"{self.test_samples_dir}/multiple_tagged_sentences.bio"

        from_tagged_file_to_bio_file(
            multiple_sentences__file, output_file, tag_entities_pairs
        )

        with open(expected_file, "r") as f:
            expected = f.read()

        with open(output_file, "r") as f:
            result = f.read()

        self.assertEqual(result, expected)

    def test_from_bio_file_to_examples(self):
        bio_file = f"{self.test_samples_dir}/multiple_tagged_sentences.bio"
        text_file = f"{self.test_samples_dir}/multiple_tagged_sentences.txt"

        examples, labels, vocab, unique_labels = from_bio_file_to_examples(bio_file)

        with open(text_file, "r") as f:
            content = f.read()
            lines = content.split("\n")

            # The number of inputs must be equal to the number of lines before parsing
            self.assertEqual(len(examples[0]), len(lines))
            self.assertEqual(len(examples[1]), len(lines))

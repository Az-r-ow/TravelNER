import unittest
from pathlib import Path
from travel_resolver.libs.nlp.ner.data_processing import (
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

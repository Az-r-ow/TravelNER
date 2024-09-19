import unittest
from travel_resolver.libs.nlp.langage_detection.extractor import (
    extract_data_from_string,
)


class TestExtractor(unittest.TestCase):

    def test_correct_extraction(self):
        input = "aabccooeeeeyln"
        result = extract_data_from_string(input)
        self.assertEqual(
            result,
            [
                25.0,
                12.5,
                25.0,
                0.0,
                50.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                12.5,
                0.0,
                12.5,
                25.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                12.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        input2 = "aabccooeeeeylnñßãç"
        result2 = extract_data_from_string(input2)
        self.assertEqual(
            result2,
            [
                16.67,
                8.33,
                16.67,
                0.0,
                33.33,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.33,
                0.0,
                8.33,
                16.67,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.33,
                0.0,
                5.56,
                5.56,
                5.56,
                5.56,
            ],
        )

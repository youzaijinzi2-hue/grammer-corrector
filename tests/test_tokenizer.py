import unittest

from grammer_corrector.tokenizer import build_vocabulary, decode, encode, tokenize


class TokenizeTests(unittest.TestCase):
    def test_splits_words_and_punctuation(self):
        self.assertEqual(
            tokenize("He go to school."),
            ["He", "go", "to", "school", "."],
        )

    def test_preserves_contractions_and_skips_whitespace(self):
        self.assertEqual(tokenize("  I\tdon't\nknow!  "), ["I", "don't", "know", "!"])

    def test_empty_and_whitespace_only_input(self):
        for text in ("", " \t\n"):
            with self.subTest(text=text):
                self.assertEqual(tokenize(text), [])


class VocabularyTests(unittest.TestCase):
    def test_assigns_stable_ids_to_unique_tokens(self):
        texts = ["He go.", "He goes."]
        expected = {
            "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
            ".": 4, "He": 5, "go": 6, "goes": 7,
        }
        self.assertEqual(build_vocabulary(texts), expected)
        self.assertEqual(build_vocabulary(texts[::-1]), expected)

    def test_empty_corpus_keeps_special_tokens(self):
        self.assertEqual(
            build_vocabulary([]),
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
        )


class EncodeTests(unittest.TestCase):
    def test_encodes_tokens_with_start_and_end_markers(self):
        vocabulary = build_vocabulary(["He go.", "He goes."])
        self.assertEqual(encode("He go.", vocabulary), [1, 5, 6, 4, 2])

    def test_encodes_unknown_tokens(self):
        vocabulary = build_vocabulary(["He go."])
        self.assertEqual(encode("He runs.", vocabulary), [1, 5, 3, 4, 2])

    def test_encodes_empty_input(self):
        vocabulary = build_vocabulary([])
        self.assertEqual(encode("", vocabulary), [1, 2])


class DecodeTests(unittest.TestCase):
    def test_restores_words_and_punctuation(self):
        text = "Yes, he goes."
        vocabulary = build_vocabulary([text])
        self.assertEqual(decode(encode(text, vocabulary), vocabulary), text)

    def test_skips_start_and_padding_and_stops_at_end(self):
        vocabulary = build_vocabulary(["He goes."])
        token_ids = [0, 1, vocabulary["He"], 0, 2, vocabulary["goes"]]
        self.assertEqual(decode(token_ids, vocabulary), "He")

    def test_displays_unknown_tokens(self):
        vocabulary = build_vocabulary([])
        self.assertEqual(decode([1, 3, 999, 2], vocabulary), "<unk> <unk>")

    def test_decodes_empty_sentence(self):
        vocabulary = build_vocabulary([])
        self.assertEqual(decode([1, 2], vocabulary), "")

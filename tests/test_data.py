from grammer_corrector.data import encode_pairs, generate_agreement_pairs
from grammer_corrector.tokenizer import build_vocabulary


def test_generates_agreement_pairs():
    assert generate_agreement_pairs() == [
        ("He go to school.", "He goes to school."),
        ("He study at home.", "He studies at home."),
        ("He play outside.", "He plays outside."),
        ("She go to school.", "She goes to school."),
        ("She study at home.", "She studies at home."),
        ("She play outside.", "She plays outside."),
    ]


def test_generates_pairs_in_the_same_order_each_time():
    assert generate_agreement_pairs() == generate_agreement_pairs()


def test_encodes_both_sides_of_generated_pairs():
    pairs = generate_agreement_pairs()
    vocabulary = build_vocabulary([text for pair in pairs for text in pair])

    encoded = encode_pairs(pairs, vocabulary)

    assert len(encoded) == 6
    assert encoded[0] == (
        [1, 5, 8, 17, 14, 4, 2],
        [1, 5, 9, 17, 14, 4, 2],
    )
    assert encoded[-1] == (
        [1, 6, 12, 11, 4, 2],
        [1, 6, 13, 11, 4, 2],
    )

from grammer_corrector import evaluation
from grammer_corrector.data import generate_agreement_pairs
from grammer_corrector.tokenizer import build_vocabulary, encode


def test_evaluation_examples_are_unseen_but_in_vocabulary():
    training_sentences = {text for pair in generate_agreement_pairs() for text in pair}
    vocabulary = build_vocabulary(list(training_sentences))
    for source, expected in evaluation.AGREEMENT_EVALUATION_PAIRS:
        assert source not in training_sentences
        assert expected not in training_sentences
        assert vocabulary["<unk>"] not in encode(source, vocabulary)
        assert vocabulary["<unk>"] not in encode(expected, vocabulary)


def test_evaluation_generates_from_source_and_compares_with_expected(monkeypatch):
    vocabulary = build_vocabulary(["He go.", "He goes.", "She goes."])
    model = object()

    def generate(actual_model, source_ids, actual_vocabulary, max_new_tokens):
        assert actual_model is model
        assert actual_vocabulary is vocabulary
        assert max_new_tokens == 8
        if source_ids == encode("He go.", vocabulary):
            return encode("He goes.", vocabulary)
        assert source_ids == encode("She goes.", vocabulary)
        return encode("He goes.", vocabulary)

    monkeypatch.setattr(evaluation, "greedy_decode", generate)
    results = evaluation.evaluate_pairs(
        model, [("He go.", "He goes."), ("She goes.", "She goes.")], vocabulary, 8
    )
    assert [result.prediction for result in results] == ["He goes.", "He goes."]
    assert [result.exact_match for result in results] == [True, False]


def test_expanded_training_adds_combinations_without_evaluation_overlap():
    baseline = generate_agreement_pairs()
    expanded = generate_agreement_pairs(expanded=True)
    assert len(set(expanded)) == 12
    assert set(baseline) < set(expanded)
    assert ("He go home.", "He goes home.") in expanded
    assert ("She study at school.", "She studies at school.") in expanded
    assert ("He play at school.", "He plays at school.") in expanded
    training_texts = {text for pair in expanded for text in pair}
    evaluation_texts = {text for pair in evaluation.AGREEMENT_EVALUATION_PAIRS for text in pair}
    assert training_texts.isdisjoint(evaluation_texts)
    assert build_vocabulary(list(training_texts)) == build_vocabulary(
        [text for pair in baseline for text in pair]
    )


def test_identity_training_preserves_targets_without_evaluation_overlap():
    correction_pairs = generate_agreement_pairs(expanded=True)
    all_pairs = generate_agreement_pairs(expanded=True, include_identity=True)

    assert all_pairs == correction_pairs + [(target, target) for _, target in correction_pairs]
    evaluation_texts = {text for pair in evaluation.AGREEMENT_EVALUATION_PAIRS for text in pair}
    assert evaluation_texts.isdisjoint(text for pair in all_pairs for text in pair)

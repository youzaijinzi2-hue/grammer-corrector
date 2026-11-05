from grammer_corrector.data import generate_agreement_pairs
from grammer_corrector.data_experiment import EXTRA_EVALUATION, training_pool
from grammer_corrector.evaluation import AGREEMENT_EVALUATION_PAIRS


def test_training_pool_is_nested_reproducible_and_disjoint_from_evaluation():
    pool = training_pool()
    assert pool == training_pool()
    assert pool[:12] == generate_agreement_pairs(expanded=True)
    assert len(set(pool[:100])) == 100
    held_out = {text for pair in (*AGREEMENT_EVALUATION_PAIRS, *EXTRA_EVALUATION) for text in pair}
    assert held_out.isdisjoint(text for pair in pool for text in pair)

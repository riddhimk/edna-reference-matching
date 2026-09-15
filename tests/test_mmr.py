import numpy as np

from src.mmr import mmr_select, _normalize_relevance


def test_normalize_relevance_min_max():
    assert _normalize_relevance([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_normalize_relevance_degenerate_all_equal():
    assert _normalize_relevance([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_normalize_relevance_single_value():
    assert _normalize_relevance([3.0]) == [1.0]


def test_mmr_select_empty_input():
    assert mmr_select([], [], np.zeros((0, 4)), lam=0.8, k_results=10) == []


def test_mmr_select_single_candidate():
    emb = np.array([[1.0, 0.0]])
    result = mmr_select([42], [5.0], emb, lam=0.8, k_results=10)
    assert result == [42]


def test_mmr_first_pick_is_highest_relevance_for_positive_lambda():
    # orthonormal embeddings -> similarity 0 between any distinct pair,
    # so the diversity term is 0 for everyone on the first pick
    # regardless of lambda; only relevance can break the tie.
    emb = np.eye(3)
    doc_idxs = [10, 20, 30]
    relevance = [0.1, 0.9, 0.5]
    result = mmr_select(doc_idxs, relevance, emb, lam=0.5, k_results=3)
    assert result[0] == 20


def test_mmr_lambda_1_is_pure_relevance_ranking():
    emb = np.eye(3)
    doc_idxs = [10, 20, 30]
    relevance = [0.1, 0.9, 0.5]
    result = mmr_select(doc_idxs, relevance, emb, lam=1.0, k_results=3)
    assert result == [20, 30, 10]  # sorted purely by relevance, descending


def test_mmr_prefers_diversity_over_a_near_duplicate_with_higher_relevance():
    # A is most relevant and picked first. B is a near-duplicate of A
    # with the second-highest relevance; C is distinct but low relevance.
    # A moderate, diversity-leaning lambda should still pick C over the
    # near-duplicate B for the second slot.
    emb = np.array([
        [1.0, 0.0, 0.0],    # A
        [0.99, 0.01, 0.0],  # B -- near-duplicate of A
        [0.0, 1.0, 0.0],    # C -- distinct
    ])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    doc_idxs = ["A", "B", "C"]
    relevance = [1.0, 0.9, 0.1]
    result = mmr_select(doc_idxs, relevance, emb, lam=0.3, k_results=3)
    assert result[0] == "A"
    assert result[1] == "C"


def test_mmr_deterministic_output():
    emb = np.eye(4)
    doc_idxs = [1, 2, 3, 4]
    relevance = [0.5, 0.5, 0.9, 0.1]
    r1 = mmr_select(doc_idxs, relevance, emb, lam=0.6, k_results=4)
    r2 = mmr_select(doc_idxs, relevance, emb, lam=0.6, k_results=4)
    assert r1 == r2


def test_mmr_respects_k_results_cap():
    emb = np.eye(5)
    doc_idxs = [1, 2, 3, 4, 5]
    relevance = [0.5] * 5
    result = mmr_select(doc_idxs, relevance, emb, lam=0.7, k_results=2)
    assert len(result) == 2


def test_mmr_k_results_larger_than_input_returns_all():
    emb = np.eye(2)
    doc_idxs = [1, 2]
    relevance = [0.5, 0.9]
    result = mmr_select(doc_idxs, relevance, emb, lam=0.7, k_results=10)
    assert len(result) == 2
    assert set(result) == {1, 2}


def test_mmr_output_is_permutation_of_input_no_duplicates():
    emb = np.random.RandomState(0).randn(6, 8)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    doc_idxs = list(range(6))
    relevance = [0.9, 0.1, 0.5, 0.7, 0.3, 0.6]
    result = mmr_select(doc_idxs, relevance, emb, lam=0.5, k_results=6)
    assert sorted(result) == sorted(doc_idxs)
    assert len(set(result)) == len(result)

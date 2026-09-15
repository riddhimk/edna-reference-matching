import pytest

from src.rrf_fusion import rrf_scores, fuse_top_k


def test_rrf_scores_single_list():
    scores = rrf_scores([[10, 20, 30]], k_const=0)
    assert scores[10] == 1.0        # 1/(0+1)
    assert scores[20] == 0.5        # 1/(0+2)
    assert scores[30] == pytest.approx(1 / 3)


def test_rrf_scores_combines_two_lists():
    # doc 5 is rank1 in list A and rank2 in list B
    scores = rrf_scores([[5, 6], [7, 5]], k_const=0)
    assert scores[5] == pytest.approx(1.0 + 0.5)
    assert scores[6] == pytest.approx(0.5)
    assert scores[7] == pytest.approx(1.0)


def test_rrf_scores_absent_doc_not_present_not_zero():
    scores = rrf_scores([[1, 2, 3]], k_const=0)
    assert 999 not in scores


def test_fuse_top_k_orders_by_combined_score():
    # doc1: 1/1 + 1/2 = 1.5; doc2: 1/2 + 1/3 = 0.833; doc3: 1/3 + 1/1 = 1.333
    ranked = fuse_top_k([[1, 2, 3], [3, 1, 2]], k_const=0, k_results=3)
    assert ranked == [1, 3, 2]


def test_fuse_top_k_respects_k_results_cap():
    ranked = fuse_top_k([[1, 2, 3, 4, 5]], k_const=10, k_results=2)
    assert len(ranked) == 2


def test_fuse_top_k_handles_empty_list():
    ranked = fuse_top_k([[], [1, 2]], k_const=10, k_results=5)
    assert ranked == [1, 2]


def test_higher_k_const_flattens_rank_differences():
    small_k = rrf_scores([[1, 2]], k_const=0)
    big_k = rrf_scores([[1, 2]], k_const=1000)
    ratio_small = small_k[1] / small_k[2]
    ratio_big = big_k[1] / big_k[2]
    assert ratio_small > ratio_big  # small k_const preserves the rank gap more strongly


def test_candidate_in_both_lists_outscores_single_list_candidate_at_same_rank():
    scores = rrf_scores([[1, 9], [1, 9]], k_const=10)  # doc1 rank1 in both, doc9 rank2 in both
    other = rrf_scores([[5, 9]], k_const=10)  # doc5 rank1 in one list only
    assert scores[1] > other[5]

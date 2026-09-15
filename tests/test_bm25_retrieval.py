from src.bm25_retrieval import kmers, BM25Index


def test_kmers_basic():
    assert kmers("ACGTAC", 3) == ["ACG", "CGT", "GTA", "TAC"]


def test_kmers_case_insensitive():
    assert kmers("acgtac", 3) == kmers("ACGTAC", 3)


def test_kmers_skips_windows_with_ambiguous_base():
    # windows of "ACGNAC" at k=3: ACG, CGN, GNA, NAC -- only ACG is pure ACGT
    assert kmers("ACGNAC", 3) == ["ACG"]


def test_kmers_too_short_returns_empty():
    assert kmers("AC", 3) == []


def test_kmers_exact_length_returns_one_window():
    assert kmers("ACG", 3) == ["ACG"]


def test_bm25_index_build_basic_structure():
    ids = ["r1", "r2"]
    seqs = ["ACGTACGT", "TTTTTTTT"]
    idx = BM25Index(k=4)
    idx.build(ids, seqs)
    assert idx.n_docs == 2
    assert idx.doc_len == [5, 5]  # 8 - 4 + 1 = 5 overlapping windows each
    assert idx.avg_doc_len == 5.0
    assert idx.postings["TTTT"][1] == 5  # every window of r2 is "TTTT"


def test_bm25_scores_exact_match_top_ranked():
    ids = ["match", "decoy"]
    seqs = ["ACGTACGTACGT", "GGGGCCCCAAAA"]
    idx = BM25Index(k=5)
    idx.build(ids, seqs)
    ranked = idx.top_k("ACGTACGTACGT", k_results=2)
    assert ranked[0][0] == 0  # "match" doc ranks first
    assert ranked[0][1] > 0


def test_bm25_top_k_empty_when_no_shared_kmers():
    idx = BM25Index(k=4)
    idx.build(["a"], ["AAAAAAAA"])
    assert idx.top_k("CCCCCCCC", k_results=5) == []


def test_bm25_top_k_never_exceeds_requested_size():
    ids = ["a", "b", "c"]
    seqs = ["ACGTACGTACGT"] * 3
    idx = BM25Index(k=4)
    idx.build(ids, seqs)
    ranked = idx.top_k("ACGTACGTACGT", k_results=2)
    assert len(ranked) == 2


def test_bm25_idf_zero_for_term_absent_from_corpus():
    idx = BM25Index(k=4)
    idx.build(["a"], ["ACGTACGT"])
    assert idx._idf("ZZZZ") == 0.0


def test_bm25_idf_higher_for_rarer_term():
    ids = ["a", "b", "c"]
    seqs = ["AAAACCCC", "AAAAGGGG", "AAAATTTT"]  # "AAAA" in all 3 docs, "CCCC" in only 1
    idx = BM25Index(k=4)
    idx.build(ids, seqs)
    assert idx._idf("CCCC") > idx._idf("AAAA")


def test_bm25_build_excludes_docs_not_passed_in():
    # simulates the leakage-prevention contract: a doc never passed to
    # build() must be structurally unreachable from the index.
    idx = BM25Index(k=4)
    idx.build(["train_only"], ["ACGTACGT"])
    assert idx.n_docs == 1
    assert idx.doc_ids == ["train_only"]

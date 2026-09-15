import csv
import numpy as np
import pytest

from src.final_pipeline import PipelineIndex, assign, assign_batch


def _write_reference_table(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reference_ID", "Sequence"])
        for rid, seq in rows:
            w.writerow([rid, seq])


def _write_ids_csv(path, ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "Reference_ID"])
        for i, rid in enumerate(ids):
            w.writerow([i, rid])


def _make_index(tmp_path):
    refs = [
        ("ref_match", "ACGTACGTACGTACGTACGTACGTACGTACGT"),
        ("ref_decoy1", "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"),
        ("ref_decoy2", "TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG"),
        ("ref_near_dup", "ACGTACGTACGTACGTACGTACGTACGTACGA"),  # 1bp different from ref_match
    ]
    ref_table_csv = tmp_path / "ref_table.csv"
    ids_csv = tmp_path / "ref_ids.csv"
    emb_npy = tmp_path / "ref_emb.npy"

    _write_reference_table(ref_table_csv, refs)
    ids = [r[0] for r in refs]
    _write_ids_csv(ids_csv, ids)

    rng = np.random.RandomState(0)
    emb = rng.randn(len(refs), 8).astype("float32")
    np.save(emb_npy, emb)

    return PipelineIndex.build(str(ref_table_csv), str(emb_npy), str(ids_csv))


def test_assign_finds_the_exact_matching_reference(tmp_path):
    index = _make_index(tmp_path)
    query = "ACGTACGTACGTACGTACGTACGTACGTACGT"  # identical to ref_match
    result = assign(query, index)
    assert result["Nearest_Reference_ID"] == "ref_match"
    assert result["identity_pct"] == 100.0
    assert result["Confidence_Decision"] == "assigned"


def test_assign_returns_unclassified_when_no_kmers_shared(tmp_path):
    index = _make_index(tmp_path)
    query = "N" * 33  # ambiguous base -> kmers() skips every window -> no BM25 candidates at all
    result = assign(query, index)
    assert result["Confidence_Decision"] == "unclassified"
    assert result["Nearest_Reference_ID"] == ""


def test_assign_returns_best_guess_when_candidate_found_but_below_threshold(tmp_path):
    index = _make_index(tmp_path)
    # 5 mutations spread across ref_decoy1 (~84% identity): shares several 6-mers
    # with it (so BM25/MMR still surface it as a candidate) but won't reach the
    # frozen 95% identity bar on realignment.
    query = "GGTGCCCCAGAATTTTAGGGCCCGAAAATCTT"
    result = assign(query, index)
    assert result["Confidence_Decision"] == "best_guess"
    assert result["Nearest_Reference_ID"] != ""
    assert 75.0 <= result["identity_pct"] < 95.0


def test_assign_returns_best_guess_low_confidence_when_match_is_weak(tmp_path):
    index = _make_index(tmp_path)
    # heavily mutated relative to every reference (~64% identity to its
    # nearest hit) -- still finds a BM25 candidate but it's a weak match.
    query = "AAAAAAAACCAATTTTAAAAAAAACCCCAAAA"
    result = assign(query, index)
    assert result["Confidence_Decision"] == "best_guess_low_confidence"
    assert result["Nearest_Reference_ID"] != ""
    assert 0.0 < result["identity_pct"] < 75.0


def test_assign_batch_writes_csv_with_expected_rows(tmp_path):
    index = _make_index(tmp_path)
    read_ids = ["r1", "r2"]
    read_seqs = ["ACGTACGTACGTACGTACGTACGTACGTACGT", "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"]
    out_csv = tmp_path / "results.csv"
    results = assign_batch(read_ids, read_seqs, index, out_csv=str(out_csv))

    assert len(results) == 2
    assert out_csv.exists()
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["Read_ID"] == "r1"
    assert rows[1]["Read_ID"] == "r2"


def test_assign_is_deterministic(tmp_path):
    index = _make_index(tmp_path)
    query = "ACGTACGTACGTACGTACGTACGTACGTACGT"
    r1 = assign(query, index)
    r2 = assign(query, index)
    assert r1 == r2


def test_pipeline_index_build_validates_id_order(tmp_path):
    refs = [("a", "ACGTACGT"), ("b", "TTTTGGGG")]
    ref_table_csv = tmp_path / "ref_table.csv"
    ids_csv = tmp_path / "ref_ids.csv"
    emb_npy = tmp_path / "ref_emb.npy"
    _write_reference_table(ref_table_csv, refs)
    _write_ids_csv(ids_csv, ["b", "a"])  # deliberately mismatched order
    np.save(emb_npy, np.zeros((2, 4), dtype="float32"))

    with pytest.raises(AssertionError, match="order"):
        PipelineIndex.build(str(ref_table_csv), str(emb_npy), str(ids_csv))

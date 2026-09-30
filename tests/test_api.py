"""
Offline tests for the results API.

These use FastAPI's TestClient, so no server and no network are needed. They
assert the API's CONTRACT -- status codes, response shape, filter behaviour --
not specific metric values, so re-running an experiment cannot break them.

Tests that depend on a result file are skipped when that file is absent, so the
suite still passes on a checkout without the (gitignored) data/ directory.
"""
import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import ROOT, PRF_FILES, READ_DATASETS, app  # noqa: E402

client = TestClient(app)


def _has(rel):
    return os.path.exists(os.path.join(ROOT, rel))


def test_index_lists_endpoints():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["inference"] is False
    assert "GET /metrics/prf" in body["endpoints"]


def test_health_reports_availability():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    for section in ("precision_recall_f1", "architecture_evaluation", "read_datasets"):
        assert section in body["available"]


def test_kingdoms_shape():
    body = client.get("/kingdoms").json()
    for k in ("fish", "bacteria", "eukaryota_large"):
        assert k in body


def test_unknown_kingdom_is_404():
    assert client.get("/metrics/prf?kingdom=bogus").status_code == 404


def test_unknown_dataset_is_404():
    assert client.get("/reads/nonexistent").status_code == 404


def test_unknown_model_is_404():
    assert client.get("/models/nosuchmodel").status_code == 404


@pytest.mark.skipif(not _has(PRF_FILES["fish"]), reason="PRF results not present")
def test_prf_filters_by_kingdom_and_size():
    body = client.get("/metrics/prf?kingdom=fish&n=5000").json()
    assert body["count"] == 1
    kd = body["kingdoms"][0]
    assert kd["kingdom"] == "fish"
    assert {r["n"] for r in kd["results"]} == {5000}
    assert {r["arm"] for r in kd["results"]} == {"frozen", "finetuned"}
    assert "fully_independent_samples" in kd["split"]


@pytest.mark.skipif(not _has(PRF_FILES["fish"]), reason="PRF results not present")
def test_prf_metrics_are_internally_consistent():
    """precision/recall must follow from the reported TP/FP/FN."""
    for kd in client.get("/metrics/prf").json()["kingdoms"]:
        for r in kd["results"]:
            tp, fp, fn = r["tp"], r["fp"], r["fn"]
            assert r["precision"] == pytest.approx(tp / (tp + fp), abs=1e-6)
            assert r["recall"] == pytest.approx(tp / (tp + fn), abs=1e-6)
            assert tp + fp + fn == r["n"]


@pytest.mark.skipif(not _has(PRF_FILES["fish"]), reason="PRF results not present")
def test_prf_comparison_delta_matches_arms():
    body = client.get("/metrics/prf/comparison?n=1000").json()
    for k in body["kingdoms"]:
        assert k["delta"]["correct_calls"] == k["finetuned"]["tp"] - k["frozen"]["tp"]


def test_prf_comparison_unknown_size_is_404():
    assert client.get("/metrics/prf/comparison?n=999").status_code == 404


def test_models_listing():
    body = client.get("/models").json()
    assert body["count"] == len(body["models"])
    for m in body["models"]:
        assert "name" in m


_READ_SLUG = "tara_bacteria_16s_v4v5"


@pytest.mark.skipif(not _has(READ_DATASETS[_READ_SLUG]["path"]), reason="read results not present")
def test_reads_pagination_and_filtering():
    all_rows = client.get(f"/reads/{_READ_SLUG}?limit=1").json()
    total = all_rows["total_matching"]
    assert len(all_rows["reads"]) == 1

    # A decision filter can only narrow the set.
    assigned = client.get(f"/reads/{_READ_SLUG}?decision=assigned&limit=1").json()
    assert assigned["total_matching"] <= total
    assert all(r["Confidence_Decision"] == "assigned" for r in assigned["reads"])

    # offset must shift the window
    first = client.get(f"/reads/{_READ_SLUG}?limit=1&offset=0").json()["reads"][0]
    second = client.get(f"/reads/{_READ_SLUG}?limit=1&offset=1").json()["reads"][0]
    assert first != second


@pytest.mark.skipif(not _has(READ_DATASETS[_READ_SLUG]["path"]), reason="read results not present")
def test_reads_min_score_filter_is_respected():
    body = client.get(f"/reads/{_READ_SLUG}?min_score=99&limit=50").json()
    col = body["score_column"]
    assert col is not None
    assert all(float(r[col]) >= 99 for r in body["reads"])


@pytest.mark.skipif(not _has(READ_DATASETS[_READ_SLUG]["path"]), reason="read results not present")
def test_reads_summary_counts_add_up():
    body = client.get(f"/reads/{_READ_SLUG}/summary").json()
    assert sum(body["decisions"].values()) == body["n_reads"]
    assert sum(t["reads"] for t in body["top"]) <= body["n_reads"]


def test_datasets_listing_declares_columns():
    for d in client.get("/reads/datasets").json()["datasets"]:
        assert "dataset" in d and "kingdom" in d and "part" in d
        if d.get("n_reads"):
            assert d["columns"], f"{d['dataset']} reported rows but no columns"

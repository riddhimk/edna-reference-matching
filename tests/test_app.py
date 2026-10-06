"""
Offline tests for uploads, run history, analytics and the chatbot loop.

The pipeline runs for real, against a tiny synthetic reference set built in a
temp directory, so these need neither data/ nor network nor an API key. The
chatbot is driven by a scripted fake client: the tests check the tool loop,
history persistence and error handling, not what a model would say.
"""
import gzip
import hashlib
import io
import random
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from src import app_routes, chat, run_analysis as ra  # noqa: E402
from src.api import app  # noqa: E402
from src.run_store import RunStore  # noqa: E402
from src.upload_io import UploadError, parse_upload  # noqa: E402

rng = random.Random(7)


def _seq(n):
    return "".join(rng.choice("ACGT") for _ in range(n))


REFS = {f"REF{i}": _seq(180) for i in range(6)}


def _fastq(reads):
    return "".join(f"@{rid} extra\n{s}\n+\n{'I' * len(s)}\n" for rid, s in reads).encode()


def _reads():
    # 4 exact substrings of references (-> assigned), 2 copies of one of them
    # (exercises identical-read collapsing), and 2 unrelated sequences.
    out = [(f"r{i}", REFS[f"REF{i}"][10:130]) for i in range(4)]
    out += [("r_dup", REFS["REF0"][10:130]), ("r_noise1", _seq(120)), ("r_noise2", _seq(120))]
    return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    table = tmp_path / "refs.csv"
    with open(table, "w") as f:
        f.write("Reference_ID,Taxonomy_Full,Organism_Label,Sequence\n")
        for i, (rid, s) in enumerate(REFS.items()):
            f.write(f"{rid},Bacteria;Phylum{i % 2};Genus{i},Species {i},{s}\n")
    np.save(tmp_path / "emb.npy", np.random.default_rng(0).normal(size=(len(REFS), 8)).astype("float32"))
    with open(tmp_path / "ids.csv", "w") as f:
        f.write("Reference_ID\n" + "".join(f"{r}\n" for r in REFS))

    marker = ra.Marker(key="test_marker", label="Test", kingdom="bacteria", description="synthetic",
                       validated_part="tests", validated_accession="none", validated_dataset="none", ref_table=str(table), ref_embeddings=str(tmp_path / "emb.npy"),
                       ref_ids=str(tmp_path / "ids.csv"))
    monkeypatch.setitem(ra.MARKERS, "test_marker", marker)
    ra.load_index.cache_clear()
    ra.load_taxonomy.cache_clear()
    monkeypatch.setenv("EDNA_WORKERS", "1")
    app_routes.state.configure(str(tmp_path / "app"))
    yield TestClient(app)
    app_routes.state.store = None
    ra.load_index.cache_clear()
    ra.load_taxonomy.cache_clear()


def _upload(client, data: bytes, name="reads.fastq", marker="test_marker"):
    return client.post("/uploads", files={"file": (name, io.BytesIO(data))}, data={"marker": marker})


def _wait(client, run_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


# ------------------------------------------------------------- parsing ---

def test_parse_fastq_fasta_and_gzip_share_content_hash():
    fq = _fastq([("a", "ACGTN"), ("b", "acgu")])
    p = parse_upload(fq)
    assert p.file_format == "fastq" and p.read_ids == ["a", "b"] and p.sequences == ["ACGTN", "ACGT"]
    gz = parse_upload(gzip.compress(fq))
    assert gz.compressed and gz.content_sha256 == p.content_sha256 and gz.raw_sha256 != p.raw_sha256

    fa = parse_upload(b">x desc\nACG\nTTA\n>y\nGG\n")
    assert fa.file_format == "fasta" and fa.read_ids == ["x", "y"] and fa.sequences == ["ACGTTA", "GG"]


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"\n\n", "empty"),
    (gzip.compress(b""), "empty"),
    (b"hello\n", "not a FASTQ"),
    (b"@a\nACGT\n+\nIIIII\n", "quality string length"),
    (b"@a\nACGT\n+\nII\n", "quality string is complete"),
    (b"@a\nACGT\n", "separator"),
    (b"@a\nAC1T\n+\nIIII\n", "not nucleotide codes"),
    (b"\x1f\x8bnot really gzip", "could not be decompressed"),
    (b"\xff\xfe@a\n", "non-text bytes"),
])
def test_parse_rejects_only_empty_or_malformed(data, msg):
    with pytest.raises(UploadError, match=msg):
        parse_upload(data)


@pytest.mark.parametrize("data,ids,seqs", [
    (b"@a\nAC\nGT\n+\nII\nII\n", ["a"], ["ACGT"]),                          # wrapped FASTQ
    (b"@a\r\nacgu\r\n+\r\nIIII\r\n", ["a"], ["ACGT"]),                       # CRLF, lower case, RNA
    (b"@a\nACGT\n+\nIIII\n\n\n@b\nAC\n+\nII\n\n", ["a", "b"], ["ACGT", "AC"]),  # blank lines between
    (b"@a\n\n+\n\n@b\nAC\n+\nII\n", ["a", "b"], ["", "AC"]),                 # an empty read
    (b"@a\nACGT\n+\n@III\n", ["a"], ["ACGT"]),                                 # quality starting with '@'
    (b";comment\n>x\nAC-G\nT.T\n", ["x"], ["ACGTT"]),                            # FASTA comment and gaps
])
def test_parse_accepts_real_world_variants(data, ids, seqs):
    p = parse_upload(data)
    assert (p.read_ids, p.sequences) == (ids, seqs)


def test_parse_has_no_read_limit():
    reads = [(f"r{i}", "ACGT" * 30) for i in range(250_000)]
    assert len(parse_upload(_fastq(reads)).sequences) == 250_000


# --------------------------------------------------------------- store ---

def test_store_duplicates_are_per_marker_and_failed_runs_do_not_block(tmp_path):
    s = RunStore(str(tmp_path / "db.sqlite"))
    base = dict(raw_sha256="r", content_sha256="c", filename="f", size_bytes=1, file_format="fastq", stored_path="p")
    run, created = s.create_run_unless_duplicate(marker="m1", **base)
    assert created
    assert s.create_run_unless_duplicate(marker="m1", **base) == (s.get_run(run["id"]), False)
    assert s.create_run_unless_duplicate(marker="m2", **base)[1]          # other marker: a real new analysis
    s.update_run(run["id"], status="failed")
    assert s.create_run_unless_duplicate(marker="m1", **base)[1]          # failed run: retry allowed


def test_store_marks_interrupted_runs_failed(tmp_path):
    path = str(tmp_path / "db.sqlite")
    s = RunStore(path)
    run, _ = s.create_run_unless_duplicate(marker="m", raw_sha256="r", content_sha256="c", filename="f",
                                           size_bytes=1, file_format="fastq", stored_path="p")
    assert RunStore(path).fail_interrupted_runs() == 1
    assert "restart" in s.get_run(run["id"])["error"]


# ------------------------------------------------------- upload -> run ---

def test_upload_runs_pipeline_and_reports_findings(client):
    data = _fastq(_reads())
    r = _upload(client, data)
    assert r.status_code == 202 and r.json()["duplicate"] is False
    run = _wait(client, r.json()["run"]["id"])
    assert run["status"] == "done", run["error"]
    assert "stored_path" not in run and "results_path" not in run       # no server paths leak

    s = run["summary"]
    assert s["input"]["n_reads"] == 7
    assert s["input"]["n_unique_sequences_classified"] == 6           # the duplicate read was collapsed
    dec = {d["decision"]: d["reads"] for d in s["decisions"]}
    assert dec["assigned"] == 5 and sum(dec.values()) == 7
    assert s["taxa"]["top_assigned"][0] == {"taxon": "Species 0", "reads": 2, "pct_of_subset": 40.0,
                                            "mean_identity": 100.0}
    assert {c["id"] for c in run["charts"]} >= {"decisions", "identity_histogram", "top_assigned_taxa"}

    reads = client.get(f"/runs/{run['id']}/reads", params={"decision": "assigned", "taxon": "species 0"}).json()
    assert reads["total_matching"] == 2 and {x["Read_ID"] for x in reads["reads"]} == {"r0", "r_dup"}
    csv_text = client.get(f"/runs/{run['id']}/download").text
    assert csv_text.startswith("Read_ID,") and csv_text.count("\n") == 8


def test_duplicate_upload_is_not_rerun_even_gzipped_and_renamed(client):
    data = _fastq(_reads())
    first = _upload(client, data).json()["run"]
    _wait(client, first["id"])

    again = _upload(client, gzip.compress(data), name="other_name.fq.gz")
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert again.json()["run"]["id"] == first["id"]
    assert client.get("/runs").json()["total"] == 1

    check = client.post("/uploads/check", json={"sha256": hashlib.sha256(data).hexdigest()}).json()
    assert check["already_analysed"] and check["runs"][0]["id"] == first["id"]
    assert not client.post("/uploads/check", json={"sha256": "0" * 64}).json()["already_analysed"]


def test_delete_frees_the_file_for_a_new_run(client):
    data = _fastq(_reads())
    first = _upload(client, data).json()["run"]
    _wait(client, first["id"])
    assert client.delete(f"/runs/{first['id']}").status_code == 200
    assert client.get(f"/runs/{first['id']}").status_code == 404
    assert _upload(client, data).status_code == 202


def test_upload_errors(client):
    assert _upload(client, _fastq(_reads()), marker="nope").status_code == 400
    bad = _upload(client, b"not a read file")
    assert bad.status_code == 422 and "reads.fastq" in bad.json()["detail"]


def test_compare_and_overview(client):
    a = _upload(client, _fastq(_reads())).json()["run"]["id"]
    b = _upload(client, _fastq(_reads()[:3]), name="b.fastq").json()["run"]["id"]
    _wait(client, a)
    _wait(client, b)

    cmp_ = client.get("/analytics/compare", params={"run_ids": f"{a},{b}"}).json()
    assert [r["run_id"] for r in cmp_["runs"]] == [a, b]
    assert cmp_["taxa_overlap"][0]["shared_taxa"] == 3
    assert {c["id"] for c in cmp_["charts"]} == {"decision_mix", "taxa_by_run"}
    assert client.get("/analytics/compare", params={"run_ids": a}).status_code == 400

    ov = client.get("/analytics/overview").json()
    assert ov["uploads"]["total_runs"] == 2 and ov["uploads"]["total_reads_analysed"] == 10
    for chart in ov["charts"]:
        assert all(len(s["values"]) == len(chart["labels"]) for s in chart["series"])


# ---------------------------------------------------------------- chat ---

def _block(**kw):
    return SimpleNamespace(to_dict=lambda: dict(kw), **kw)


def _resp(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks), model="fake-model")


class FakeClient:
    """Scripted stand-in for the Anthropic client: replays `script` in order
    and records every request it was sent."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


@pytest.fixture()
def claude(monkeypatch):
    monkeypatch.setenv("EDNA_CHAT_PROVIDER", "anthropic")


@pytest.fixture()
def gemini(monkeypatch):
    monkeypatch.setenv("EDNA_CHAT_PROVIDER", "gemini")


def test_chat_runs_tools_and_persists_history(client, claude, monkeypatch):
    run_id = _upload(client, _fastq(_reads())).json()["run"]["id"]
    _wait(client, run_id)
    fake = FakeClient([
        _resp("tool_use", _block(type="text", text="Checking."),
              _block(type="tool_use", id="t1", name="get_run_summary", input={"run_id": run_id})),
        _resp("end_turn", _block(type="text", text="5 of 7 reads were assigned.")),
    ])
    monkeypatch.setattr(chat, "_client", fake)

    sid = client.post("/chat/sessions", json={"run_id": run_id}).json()["id"]
    out = client.post(f"/chat/sessions/{sid}/messages", json={"message": "How did it go?"}).json()
    assert out["reply"] == "5 of 7 reads were assigned."
    assert out["tools_used"] == [{"tool": "get_run_summary", "input": {"run_id": run_id}, "error": False}]

    first, second = fake.calls
    assert run_id in first["messages"][0]["content"]                     # run context reached the model
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and '"assignment_rate_pct"' in tool_result["content"]
    assert first["system"][-1]["cache_control"] == {"type": "ephemeral"}

    transcript = client.get(f"/chat/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in transcript] == ["user", "assistant", "assistant"]
    assert transcript[0]["text"] == "How did it go?"                     # context suffix hidden


def test_chat_tool_errors_go_back_to_the_model(client, claude, monkeypatch):
    fake = FakeClient([
        _resp("tool_use", _block(type="tool_use", id="t1", name="get_run_summary", input={"run_id": "missing"})),
        _resp("end_turn", _block(type="text", text="That run does not exist.")),
    ])
    monkeypatch.setattr(chat, "_client", fake)
    sid = client.post("/chat/sessions", json={}).json()["id"]
    out = client.post(f"/chat/sessions/{sid}/messages", json={"message": "summarise run missing"}).json()
    assert out["tools_used"][0]["error"] is True
    assert fake.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_chat_without_credentials_is_503_and_stores_nothing(client, claude, monkeypatch):
    def no_creds(messages):
        raise TypeError('"Could not resolve authentication method. Expected one of api_key ..."')
    monkeypatch.setattr(chat, "_send", no_creds)
    sid = client.post("/chat/sessions", json={}).json()["id"]
    r = client.post(f"/chat/sessions/{sid}/messages", json={"message": "hi"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.get(f"/chat/sessions/{sid}").json()["messages"] == []


def test_worker_pool_matches_serial(client):
    """The multi-process path must give the serial results. bm25_score is
    compared to 1e-9 only: BM25 sums floats in hash order, and each process
    has its own hash seed, so the last bit can differ between ANY two
    processes -- two serial runs of the original scripts differ the same way."""
    seqs = [REFS[f"REF{i % 6}"][i // 6:i // 6 + 110] for i in range(240)] + [_seq(110) for _ in range(20)]
    serial = ra.classify_unique(seqs, "test_marker", workers=1, on_progress=lambda d: None)
    pooled = ra.classify_unique(seqs, "test_marker", workers=2, on_progress=lambda d: None)
    assert len(set(seqs)) >= 200 and pooled.keys() == serial.keys()
    for seq, r in serial.items():
        p = pooled[seq]
        assert {k: v for k, v in p.items() if k != "bm25_score"} == {k: v for k, v in r.items() if k != "bm25_score"}
        assert p["bm25_score"] == pytest.approx(r["bm25_score"], rel=1e-9)


# ---------------------------------------------------------- chat: Gemini ---

def _gresp(*parts, finish="STOP"):
    from google.genai import types as g
    return g.GenerateContentResponse(model_version="fake-gemini", candidates=[
        g.Candidate(content=g.Content(role="model", parts=list(parts)), finish_reason=finish)])


def test_gemini_chat_runs_tools_and_round_trips_history(client, gemini, monkeypatch):
    from google.genai import types as g
    run_id = _upload(client, _fastq(_reads())).json()["run"]["id"]
    _wait(client, run_id)
    script = [
        _gresp(g.Part(function_call=g.FunctionCall(id="c1", name="get_run_summary", args={"run_id": run_id}),
                      thought_signature=b"sig-1")),
        _gresp(g.Part(text="thinking...", thought=True), g.Part(text="5 of 7 reads were assigned.")),
        _gresp(g.Part(text="Species 0 had the most reads.")),
    ]
    calls = []

    def fake_send(contents):
        calls.append(contents)
        return script.pop(0)

    monkeypatch.setattr(chat, "_gemini_send", fake_send)
    sid = client.post("/chat/sessions", json={"run_id": run_id}).json()["id"]
    assert client.get(f"/chat/sessions/{sid}").json()["provider"] == "gemini"

    out = client.post(f"/chat/sessions/{sid}/messages", json={"message": "How did it go?"}).json()
    assert out["reply"] == "5 of 7 reads were assigned." and out["provider"] == "gemini"
    fr = calls[1][-1].parts[0].function_response
    assert fr.id == "c1" and '"assignment_rate_pct"' in fr.response["result"]

    # Second turn: stored history is replayed, thought signature intact.
    client.post(f"/chat/sessions/{sid}/messages", json={"message": "Top species?"})
    replayed = calls[2]
    assert replayed[1].parts[0].thought_signature == b"sig-1"
    assert [m["text"] for m in client.get(f"/chat/sessions/{sid}").json()["messages"]] == [
        "How did it go?", "5 of 7 reads were assigned.", "Top species?", "Species 0 had the most reads."]


@pytest.mark.parametrize("failure", [
    ("ServerError", 503, "UNAVAILABLE"),          # overloaded
    ("ClientError", 429, "RESOURCE_EXHAUSTED"),   # this model's quota used up
    ("ClientError", 404, "NOT_FOUND"),            # model retired for this key
])
def test_gemini_falls_back_to_next_model(client, gemini, monkeypatch, failure):
    from google.genai import errors
    cls, code, status = failure
    tried = []

    class Models:
        def generate_content(self, model, contents, config):
            tried.append(model)
            if model == chat.GEMINI_MODEL:
                raise getattr(errors, cls)(code, {"error": {"code": code, "message": "x", "status": status}})
            from google.genai import types as g
            return _gresp(g.Part(text="ok"))

    monkeypatch.setattr(chat, "_gemini_client", SimpleNamespace(models=Models()))
    sid = client.post("/chat/sessions", json={}).json()["id"]
    assert client.post(f"/chat/sessions/{sid}/messages", json={"message": "hi"}).json()["reply"] == "ok"
    assert tried == [chat.GEMINI_MODEL, chat.GEMINI_FALLBACKS[0]]


def test_gemini_does_not_fall_back_on_a_bad_request(client, gemini, monkeypatch):
    from google.genai import errors
    tried = []

    class Models:
        def generate_content(self, model, contents, config):
            tried.append(model)
            raise errors.ClientError(400, {"error": {"code": 400, "message": "bad schema", "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(chat, "_gemini_client", SimpleNamespace(models=Models()))
    sid = client.post("/chat/sessions", json={}).json()["id"]
    assert client.post(f"/chat/sessions/{sid}/messages", json={"message": "hi"}).status_code == 502
    assert tried == [chat.GEMINI_MODEL]


def test_gemini_key_file_formats(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    f = tmp_path / "k.txt"
    monkeypatch.setenv("EDNA_GEMINI_KEY_FILE", str(f))
    f.write_text("key name: something\napi key: AQ.secret\n")
    assert chat.gemini_api_key() == "AQ.secret"
    f.write_text("AIzaBareKey\n")
    assert chat.gemini_api_key() == "AIzaBareKey"
    f.write_text("key name: only a label\n")
    assert chat.gemini_api_key() is None


def test_session_refuses_a_provider_switch(client, monkeypatch):
    monkeypatch.setenv("EDNA_CHAT_PROVIDER", "anthropic")
    sid = client.post("/chat/sessions", json={}).json()["id"]
    monkeypatch.setenv("EDNA_CHAT_PROVIDER", "gemini")
    r = client.post(f"/chat/sessions/{sid}/messages", json={"message": "hi"})
    assert r.status_code == 409 and "new conversation" in r.json()["detail"]


# ------------------------------------------------- nothing typed into the UI ---

def test_summary_carries_its_own_thresholds_and_flags(client):
    from src.final_pipeline import BEST_GUESS_MIN_IDENTITY, BM25_K_MER, FROZEN_IDENTITY_THRESHOLD
    run = _wait(client, _upload(client, _fastq(_reads())).json()["run"]["id"])
    s = run["summary"]
    assert s["identity_threshold"] == FROZEN_IDENTITY_THRESHOLD
    assert s["best_guess_min_identity"] == BEST_GUESS_MIN_IDENTITY
    assert s["identity"]["histogram_bin_width"] == ra.IDENTITY_BIN_WIDTH
    assert s["references"]["hubness_flag_pct"] == ra.HUBNESS_FLAG_PCT
    assert s["references"]["concentrated"] == (s["references"]["top10_share_of_assigned_pct"] >= ra.HUBNESS_FLAG_PCT)
    meaning = {d["decision"]: d["meaning"] for d in s["decisions"]}
    assert f"{FROZEN_IDENTITY_THRESHOLD:g}%" in meaning["assigned"]
    assert f"{BEST_GUESS_MIN_IDENTITY:g}" in meaning["best_guess"] and f"{BM25_K_MER}-mer" in meaning["unclassified"]


def test_markers_report_computed_reference_counts(client):
    m = next(x for x in client.get("/markers").json()["markers"] if x["key"] == "test_marker")
    assert m["n_references"] == len(REFS)
    assert m["validation"] is None                     # no saved run for the synthetic preset
    assert "validated_on" not in m


# ------------------------------------------------- automatic reference set ---

PRIMER = "GTGCCAGCAGCCGCGG"


def _trim_primer(seq):
    return seq[len(PRIMER):] if seq.startswith(PRIMER) else None


@pytest.fixture()
def two_sets(tmp_path, monkeypatch):
    """Two synthetic reference sets with unrelated sequences; set B has a primer."""
    def make(name, seqs):
        table = tmp_path / f"{name}.csv"
        with open(table, "w") as f:
            f.write("Reference_ID,Taxonomy_Full,Organism_Label,Sequence\n")
            for i, s in enumerate(seqs):
                f.write(f"{name}{i},Bacteria;P;G{i},{name} species {i},{s}\n")
        np.save(tmp_path / f"{name}.npy", np.random.default_rng(1).normal(size=(len(seqs), 8)).astype("float32"))
        with open(tmp_path / f"{name}_ids.csv", "w") as f:
            f.write("Reference_ID\n" + "".join(f"{name}{i}\n" for i in range(len(seqs))))
        return dict(ref_table=str(table), ref_embeddings=str(tmp_path / f"{name}.npy"), ref_ids=str(tmp_path / f"{name}_ids.csv"))
    r = random.Random(11)
    A = ["".join(r.choice("ACGT") for _ in range(200)) for _ in range(5)]
    B = ["".join(r.choice("ACGT") for _ in range(200)) for _ in range(5)]
    markers = {
        "set_a": ra.Marker(key="set_a", label="Set A", kingdom="x", description="", validated_part="t",
                           validated_accession="t", validated_dataset="none", **make("A", A)),
        "set_b": ra.Marker(key="set_b", label="Set B", kingdom="x", description="", validated_part="t",
                           validated_accession="t", validated_dataset="none", trim=_trim_primer, primers="P", **make("B", B)),
    }
    monkeypatch.setattr(ra, "MARKERS", markers)
    for fn in (ra.load_index, ra.load_taxonomy, ra.reference_kmers, ra.count_references):
        fn.cache_clear()
    yield A, B
    for fn in (ra.load_index, ra.load_taxonomy, ra.reference_kmers, ra.count_references):
        fn.cache_clear()


def test_detects_the_matching_set(two_sets):
    A, B = two_sets
    d = ra.detect_marker([a[i:i + 110] for a in A for i in range(0, 80, 8)])
    assert d["marker"] == "set_a" and d["confident"] and d["mode"] == "as_uploaded"
    assert [e["marker"] for e in d["evidence"]] == ["set_a", "set_b"]


def test_detects_reverse_complemented_reads(two_sets):
    A, _ = two_sets
    d = ra.detect_marker([ra.revcomp(a[i:i + 110]) for a in A for i in range(0, 80, 8)])
    assert d["marker"] == "set_a" and d["confident"]


def test_primer_reads_are_trimmed_and_pretrimmed_reads_are_kept(two_sets):
    _, B = two_sets
    with_primer = ra.detect_marker([PRIMER + b[i:i + 110] for b in B for i in range(0, 80, 8)])
    assert (with_primer["marker"], with_primer["mode"], with_primer["confident"]) == ("set_b", "primer_trim", True)
    pretrimmed = ra.detect_marker([b[i:i + 110] for b in B for i in range(0, 80, 8)])
    assert (pretrimmed["marker"], pretrimmed["mode"], pretrimmed["confident"]) == ("set_b", "as_uploaded", True)
    # used as uploaded, nothing is thrown away as "no primer"
    out = ra.run_marker("set_b", ["r1", "r2"], [B[0][:120], B[1][:120]], workers=1, mode="as_uploaded")
    assert {r["Confidence_Decision"] for r in out.rows} == {"assigned"}


def test_unrelated_reads_are_flagged_not_forced(two_sets):
    r = random.Random(5)
    d = ra.detect_marker(["".join(r.choice("ACGT") for _ in range(120)) for _ in range(50)])
    assert d["confident"] is False and d["reason"].startswith("the closest, ")


def test_auto_upload_end_to_end(client, two_sets, monkeypatch):
    A, _ = two_sets
    monkeypatch.setenv("EDNA_WORKERS", "1")
    data = _fastq([(f"r{i}", a[10:130]) for i, a in enumerate(A)] + [("empty", "")])
    up = client.post("/uploads", files={"file": ("reads.fastq", io.BytesIO(data))})     # no marker given
    assert up.status_code == 202, up.text
    run = _wait(client, up.json()["run"]["id"])
    assert run["status"] == "done", run["error"]
    assert run["marker"] == "set_a" and run["summary"]["marker_detection"]["confident"] is True
    dec = {d["decision"]: d["reads"] for d in run["summary"]["decisions"]}
    assert dec["assigned"] == 5 and dec["unclassified"] == 1           # the empty read matches nothing
    again = client.post("/uploads", files={"file": ("renamed.fq", io.BytesIO(data))})
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert client.post("/uploads/check", json={"sha256": hashlib.sha256(data).hexdigest()}).json()["already_analysed"]



def test_delete_endpoint_removes_run_locally_and_from_cloud(client, tmp_path, monkeypatch):
    """End to end through the API, with the fake rclone from the cloud tests."""
    from tests.test_cloud_backup import FAKE_RCLONE
    import stat as _stat
    rclone = tmp_path / "rclone"; rclone.write_text(FAKE_RCLONE)
    rclone.chmod(rclone.stat().st_mode | _stat.S_IEXEC)
    cloud = tmp_path / "cloud"; cloud.mkdir()
    monkeypatch.setenv("EDNA_RCLONE_BIN", str(rclone)); monkeypatch.setenv("EDNA_RCLONE_REMOTE", "fake:")
    monkeypatch.setenv("FAKE_CLOUD", str(cloud)); monkeypatch.setenv("EDNA_CLOUD_BACKUP", "on")
    from src import cloud_backup as cb
    run = _wait(client, _upload(client, _fastq(_reads())).json()["run"]["id"])
    app_dir = app_routes.state.app_dir
    assert cb.push(app_dir)["ok"]
    assert (cloud / "runs" / run["id"] / "results.csv").exists() and any((cloud / "uploads").iterdir())

    res = client.delete(f"/runs/{run['id']}").json()
    assert res["deleted"] == run["id"] and res["cloud"]["ok"], res
    assert not (cloud / "runs" / run["id"]).exists() and not any((cloud / "uploads").iterdir())
    assert not os.path.exists(os.path.join(app_dir, "runs", run["id"]))
    # the history database in the cloud is refreshed and no longer lists the run
    end = time.time() + 30
    while cb.scheduler.busy() and time.time() < end:
        time.sleep(0.1)
    import sqlite3 as _sq
    assert _sq.connect(cloud / "edna_app.db").execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0

"""
Upload, run history, results and analytics endpoints.

Everything an uploaded file goes through:

    POST /uploads/check   hash known already? -> skip the upload entirely
    POST /uploads         parse -> duplicate check -> queue a run (202)
    GET  /runs/{id}       poll status/progress; summary once done
    GET  /runs/{id}/reads per-read results, filtered and paginated
    GET  /analytics/...   chart-ready comparisons across runs and benchmarks

Runs execute one at a time on a background thread (each run already uses a
process pool across cores, so running two at once would only make both
slower). State lives in SQLite, so history survives restarts; a run cut off
by a restart is marked failed rather than left looking "in progress".

Run this API as ONE uvicorn process. The job queue is in-process; several
--workers would each mark the others' in-flight runs as interrupted.

Charts are returned as data, not images: every `charts` entry is
{id, title, type, x_label, y_label, labels, series:[{name, values}]}, so a
frontend can render any of them with one generic component.
"""
import gzip
import os
import queue
import shutil
import tempfile
import threading
import traceback
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import cloud_backup
from . import run_analysis as ra
from .run_store import RunStore, now_iso
from .upload_io import UploadError, parse_upload_file

router = APIRouter()

# There is deliberately no size or read-count limit: an upload is refused only
# when it is empty or not FASTQ/FASTA. Uploads stream to disk, never into memory.
_UPLOAD_CHUNK = 4 * 2**20


# --------------------------------------------------------------- state ---

class _State:
    """The store and job queue, created on first use so tests can point
    EDNA_APP_DIR at a temporary directory before anything touches disk."""

    def __init__(self):
        self.store: Optional[RunStore] = None
        self.app_dir: Optional[str] = None
        self.jobs: "queue.Queue" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def configure(self, app_dir: str) -> None:
        self.app_dir = app_dir
        self.store = RunStore(os.path.join(app_dir, "edna_app.db"))
        self.store.fail_interrupted_runs()

    def get_store(self) -> RunStore:
        with self.lock:
            if self.store is None:
                self.configure(os.environ.get("EDNA_APP_DIR", os.path.join(ra.ROOT, "data", "app")))
            return self.store

    def ensure_worker(self) -> None:
        with self.lock:
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=_worker_loop, name="edna-run-worker", daemon=True)
                self.worker.start()


state = _State()


def _worker_loop() -> None:
    while True:
        run_id, read_ids, sequences = state.jobs.get()
        try:
            _execute(run_id, read_ids, sequences)
        finally:
            state.jobs.task_done()


def _execute(run_id: str, read_ids: List[str], sequences: List[str]) -> None:
    store = state.get_store()
    run = store.get_run(run_id)
    if run is None:            # deleted while queued
        return
    store.update_run(run_id, status="running", started_at=now_iso())
    last = {"done": -1}

    def progress(done: int, total: int) -> None:
        if done != last["done"]:
            last["done"] = done
            store.update_run(run_id, progress_done=done, progress_total=total)

    try:
        marker, mode, detection = run["marker"], None, None
        if marker == "auto":
            # The reference set is decided from the reads themselves.
            detection = ra.detect_marker(sequences)
            marker, mode = detection["marker"], detection["mode"]
            store.update_run(run_id, marker=marker)
        out = ra.run_marker(marker, read_ids, sequences, on_progress=progress, mode=mode, detection=detection)
        path = os.path.join(state.app_dir, "runs", run_id, "results.csv")
        ra.write_results_csv(out.rows, path)
        store.update_run(run_id, status="done", finished_at=now_iso(), results_path=path, summary=out.summary)
    except Exception as e:  # a failed run must say why, never hang in "running"
        traceback.print_exc()
        store.update_run(run_id, status="failed", finished_at=now_iso(), error=f"{type(e).__name__}: {e}")
    # Copy the updated history to the cloud in the background. A backup problem
    # is recorded on /backup/status; it never affects the run itself.
    cloud_backup.scheduler.request(state.app_dir)


def _public(run: dict) -> dict:
    """A run as the API returns it: no server filesystem paths."""
    out = {k: v for k, v in run.items() if k not in ("stored_path", "results_path")}
    total = run.get("progress_total") or 0
    out["progress_pct"] = round(100 * run["progress_done"] / total, 1) if total else (100.0 if run["status"] == "done" else 0.0)
    return out


def _get_run_or_404(run_id: str) -> dict:
    run = state.get_store().get_run(run_id)
    if not run:
        raise HTTPException(404, f"no run '{run_id}'. See GET /runs")
    return run


# ------------------------------------------------------------- markers ---

@router.get("/markers", tags=["upload"], summary="Marker presets an upload can be analysed against")
def markers():
    return {"markers": ra.describe_markers(),
            "detection": {"method": f"share of the reads' {ra.DETECT_K}-mers found in each reference set, "
                                    f"over up to {ra.DETECT_SAMPLE} distinct reads, best orientation",
                          "min_score": ra.DETECT_MIN_SCORE, "min_ratio_to_runner_up": ra.DETECT_MIN_RATIO},
            "note": "Uploads are matched to a reference set automatically from the reads themselves; "
                    "reads from a gene region none of these sets covers are flagged, not forced to fit."}


# ------------------------------------------------------------- uploads ---

class UploadCheck(BaseModel):
    sha256: str
    marker: Optional[str] = None


@router.post("/uploads/check", tags=["upload"], summary="Has this file already been analysed?")
def uploads_check(body: UploadCheck):
    """Send the SHA-256 of the file's bytes (as the browser would compute it
    before uploading). If a matching run exists, skip the upload and open
    that run instead. Matches either the uploaded bytes or, for a gzip
    upload, the decompressed content."""
    store = state.get_store()
    sha = body.sha256.strip().lower()
    if body.marker and body.marker != "auto":
        found = [store.find_duplicate(body.marker, raw_sha256=sha, content_sha256=sha)]
    else:
        found = [store.find_duplicate(None, raw_sha256=sha, content_sha256=sha)]
    runs = [r for r in found if r]
    return {"sha256": sha, "already_analysed": bool(runs), "runs": [_public(r) for r in runs]}


@router.post("/uploads", tags=["upload"], summary="Upload a FASTQ/FASTA file (optionally gzipped); the reference set is detected automatically")
async def uploads(file: UploadFile = File(...), marker: str = Form("auto")):
    """marker "auto" (the default): the reference set is detected from the reads.
    Naming a marker overrides detection."""
    if marker != "auto":
        if marker not in ra.MARKERS:
            raise HTTPException(400, f"unknown marker '{marker}'. Options: auto, {', '.join(ra.MARKERS)}")
        if not ra.marker_available(ra.MARKERS[marker]):
            raise HTTPException(503, f"reference files for '{marker}' are missing on this server")
    elif not any(ra.marker_available(m) for m in ra.MARKERS.values()):
        raise HTTPException(503, "no reference set is available on this server")

    store = state.get_store()
    tmp_dir = os.path.join(state.app_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=tmp_dir, suffix=".upload")
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK):
                out.write(chunk)
                size += len(chunk)
        try:
            # Parsing a large file takes seconds; off the event loop so the
            # API keeps answering (progress polls, other pages) meanwhile.
            parsed = await run_in_threadpool(parse_upload_file, tmp)
        except UploadError as e:
            raise HTTPException(422, f"could not read '{file.filename}': {e}")
        return await run_in_threadpool(_register_upload, parsed, tmp, size, file.filename, marker)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _register_upload(parsed, tmp: str, size: int, filename: Optional[str], marker: str) -> JSONResponse:
    store = state.get_store()
    # One stored copy per distinct content, whatever it was uploaded as.
    stored = os.path.join(state.app_dir, "uploads", f"{parsed.content_sha256}.{parsed.file_format}.gz")
    run, created = store.create_run_unless_duplicate(
        raw_sha256=parsed.raw_sha256, content_sha256=parsed.content_sha256, marker=marker,
        filename=os.path.basename(filename or "upload"), size_bytes=size,
        file_format=parsed.file_format + (".gz" if parsed.compressed else ""), stored_path=stored,
        progress_total=len(parsed.sequences))

    if not created:
        return JSONResponse(status_code=200, content={
            "duplicate": True, "run": _public(run),
            "message": f"This file was already analysed against '{run['marker']}' (run {run['id']}, "
                       f"uploaded as '{run['filename']}' on {run['created_at']}). No new run was started."})

    if not os.path.exists(stored):
        os.makedirs(os.path.dirname(stored), exist_ok=True)
        if parsed.compressed:
            shutil.copyfile(tmp, stored)
        else:
            with open(tmp, "rb") as src, gzip.open(stored, "wb") as dst:
                shutil.copyfileobj(src, dst, _UPLOAD_CHUNK)
    state.ensure_worker()
    state.jobs.put((run["id"], parsed.read_ids, parsed.sequences))
    return JSONResponse(status_code=202, content={
        "duplicate": False, "run": _public(run), "n_reads": len(parsed.sequences),
        "message": f"Queued. Poll GET /runs/{run['id']} for progress."})


# ---------------------------------------------------------------- runs ---

@router.get("/runs", tags=["runs"], summary="History of analysed uploads, newest first")
def list_runs(marker: Optional[str] = None, status: Optional[str] = None,
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    runs, total = state.get_store().list_runs(marker, status, limit, offset)
    # History rows stay light: the full summary is on GET /runs/{id}.
    rows = []
    for r in runs:
        p = _public(r)
        s = p.pop("summary") or {}
        p["n_reads"] = (s.get("input") or {}).get("n_reads")
        det = s.get("marker_detection")
        p["marker_detected"] = det is not None
        p["marker_fits"] = None if det is None else det["confident"]
        p["assignment_rate_pct"] = s.get("assignment_rate_pct")
        rows.append(p)
    return {"total": total, "limit": limit, "offset": offset, "runs": rows}


@router.get("/runs/{run_id}", tags=["runs"], summary="One run: status, progress, and its full findings when done")
def get_run(run_id: str):
    run = _get_run_or_404(run_id)
    out = _public(run)
    out["charts"] = run_charts(run["summary"]) if run["summary"] else []
    return out


@router.get("/runs/{run_id}/reads", tags=["runs"], summary="Per-read results for a finished run")
def run_reads(run_id: str, decision: Optional[str] = None,
              min_identity: Optional[float] = Query(None, ge=0, le=100),
              taxon: Optional[str] = Query(None, description="case-insensitive substring of organism or taxonomy"),
              limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0)):
    run = _get_run_or_404(run_id)
    if run["status"] != "done":
        raise HTTPException(409, f"run is {run['status']}; per-read results exist once it is done")
    rows = ra.cached_results(run["results_path"])
    if decision:
        rows = [r for r in rows if r["Confidence_Decision"] == decision]
    if min_identity is not None:
        rows = [r for r in rows if (ra._num(r["identity_pct"]) or -1) >= min_identity]
    if taxon:
        t = taxon.lower()
        rows = [r for r in rows if t in r["Nearest_Organism"].lower() or t in r["Taxonomy"].lower()]
    return {"run_id": run_id, "total_matching": len(rows), "limit": limit, "offset": offset,
            "reads": rows[offset:offset + limit]}


@router.get("/runs/{run_id}/download", tags=["runs"], summary="Per-read results as CSV")
def run_download(run_id: str):
    run = _get_run_or_404(run_id)
    if run["status"] != "done":
        raise HTTPException(409, f"run is {run['status']}")
    stem = os.path.splitext(run["filename"])[0]
    return FileResponse(run["results_path"], media_type="text/csv", filename=f"{stem}_{run['marker']}_results.csv")


@router.delete("/runs/{run_id}", tags=["runs"], summary="Delete a run here and from the cloud backup")
def delete_run(run_id: str):
    """Removes the run from the history, its per-read results, and -- when no
    other run shares it -- its stored upload; then removes the same files from
    the cloud backup and refreshes the backed-up history so the run is gone
    there too. The local deletion always completes; a cloud step that fails is
    queued and retried by the next backup (see "cloud" in the response)."""
    run = _get_run_or_404(run_id)
    if run["status"] in ("queued", "running"):
        raise HTTPException(409, "cannot delete a run that is still in progress")
    store = state.get_store()
    store.delete_run(run_id)
    cloud_paths = [f"runs/{run_id}"]
    run_dir = os.path.join(state.app_dir, "runs", run_id)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    # The stored upload is shared by every run of the same content; keep it while any remain.
    upload = run.get("stored_path")
    if upload and not store.other_runs_using_file(upload, run_id):
        if os.path.exists(upload):
            os.remove(upload)
        cloud_paths.append(f"uploads/{os.path.basename(upload)}")
    cloud = cloud_backup.delete_remote(state.app_dir, cloud_paths)
    # Re-upload the history database so the cloud copy no longer lists the run.
    cloud_backup.scheduler.request(state.app_dir)
    return {"deleted": run_id, "cloud": cloud}


# ------------------------------------------------------------- backup ---

@router.get("/backup/status", tags=["backup"], summary="Cloud backup of the run history: configured? last result?")
def backup_status():
    store = state.get_store()   # makes sure state.app_dir is set
    return {**cloud_backup.check(), "enabled": cloud_backup.enabled(), "running": cloud_backup.scheduler.busy(),
            "last": cloud_backup.read_status(state.app_dir), "setup": cloud_backup.SETUP_HINT}


@router.post("/backup", tags=["backup"], summary="Back up the run history to the cloud now")
def backup_now():
    state.get_store()
    chk = cloud_backup.check()
    if not chk["configured"]:
        raise HTTPException(503, f"cloud backup is not set up: {chk['reason']}")
    cloud_backup.scheduler.request(state.app_dir)
    return {"started": cloud_backup.enabled(), "remote": chk["remote"],
            "message": "Backup started; see GET /backup/status." if cloud_backup.enabled()
            else "Automatic and on-demand backups are switched off (EDNA_CLOUD_BACKUP=off)."}


# ----------------------------------------------------------- analytics ---

def _chart(cid, title, ctype, labels, series, x_label="", y_label=""):
    return {"id": cid, "title": title, "type": ctype, "x_label": x_label, "y_label": y_label,
            "labels": labels, "series": series}


def run_charts(s: dict) -> List[dict]:
    dec = s["decisions"]
    hist = s["identity"]["histogram"]
    charts = [
        _chart("decisions", "Reads by confidence decision", "bar", [d["decision"] for d in dec],
               [{"name": "reads", "values": [d["reads"] for d in dec]}], "decision", "reads"),
        _chart("identity_histogram", f"Identity to nearest reference (threshold {s['identity_threshold']:.0f}%)",
               "histogram", [f"{b['bin_start']}-{b['bin_end']}" for b in hist],
               [{"name": "reads", "values": [b["reads"] for b in hist]}], "identity %", "reads"),
    ]
    top = s["taxa"]["top_assigned"]
    if top:
        charts.append(_chart("top_assigned_taxa", "Most-detected taxa (assigned reads)", "hbar",
                             [t["taxon"] for t in top], [{"name": "reads", "values": [t["reads"] for t in top]}],
                             "reads", "taxon"))
    lin = s["taxa"]["lineage_level_1_assigned"]
    if lin:
        charts.append(_chart("lineage_level_1", "Assigned reads by lineage level 1", "hbar",
                             [t["taxon"] for t in lin], [{"name": "reads", "values": [t["reads"] for t in lin]}],
                             "reads", "lineage"))
    return charts


@router.get("/analytics/overview", tags=["analytics"], summary="Dashboard bundle: run history plus benchmark findings")
def analytics_overview():
    from . import api   # imported here: api.py includes this router, so a top-level import would be circular

    runs, total = state.get_store().list_runs(limit=500)
    done = [r for r in runs if r["status"] == "done" and r["summary"]]
    by_status: Dict[str, int] = {}
    by_marker: Dict[str, int] = {}
    for r in runs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_marker[r["marker"]] = by_marker.get(r["marker"], 0) + 1

    run_rows = [{"run_id": r["id"], "filename": r["filename"], "marker": r["marker"], "created_at": r["created_at"],
                 "n_reads": r["summary"]["input"]["n_reads"], "assignment_rate_pct": r["summary"]["assignment_rate_pct"],
                 "median_identity_assigned": r["summary"]["identity"]["median_assigned"],
                 "n_distinct_taxa": r["summary"]["taxa"]["n_distinct_assigned"]} for r in reversed(done)]

    charts = []
    if run_rows:
        charts.append(_chart("runs_assignment_rate", "Assignment rate per uploaded run", "bar",
                             [f"{r['filename']} ({r['run_id']})" for r in run_rows],
                             [{"name": "assigned %", "values": [r["assignment_rate_pct"] for r in run_rows]}],
                             "run", "% of reads assigned"))

    # Benchmarks -- the project's measured findings, from the saved result files.
    benchmarks = {}
    try:
        prf = api.metrics_prf_comparison(5000)["kingdoms"]
        benchmarks["precision_recall_f1"] = prf
        for metric in ("precision", "recall", "f1"):
            charts.append(_chart(f"prf_{metric}", f"Held-out {metric} by kingdom, frozen vs fine-tuned (n=5,000)",
                                 "grouped_bar", [k["kingdom"] for k in prf],
                                 [{"name": arm, "values": [round(k[arm][metric], 4) for k in prf]}
                                  for arm in ("frozen", "finetuned")], "kingdom", metric))
    except HTTPException:
        pass
    try:
        arch = api.metrics_architecture(None)["kingdoms"]
        benchmarks["architecture"] = arch
        arms = [a["architecture"] for a in arch[0]["architectures"]]
        charts.append(_chart("architecture_recall_at_1", "Recall@1 by retrieval architecture", "grouped_bar", arms,
                             [{"name": k["kingdom"], "values": [round(a["recall_at_1"], 4) for a in k["architectures"]]}
                              for k in arch], "architecture", "recall@1"))
    except (HTTPException, IndexError, KeyError):
        pass
    try:
        models = api.models_comparison()["models"]
        benchmarks["embedding_models"] = models
        rec = {name: m.get("recall_at_10") for name, m in models.items() if isinstance(m.get("recall_at_10"), (int, float))}
        if rec:
            charts.append(_chart("model_recall_at_10", "Embedding model comparison: recall@10", "bar", list(rec),
                                 [{"name": "recall@10", "values": list(rec.values())}], "model", "recall@10"))
    except HTTPException:
        pass
    try:
        real = []
        for d in api.read_datasets()["datasets"]:
            if not d.get("n_reads"):
                continue
            s = api.reads_summary(d["dataset"], 5)
            n = s["n_reads"] or 1
            real.append({"dataset": d["dataset"], "kingdom": d["kingdom"], "part": d["part"],
                         "n_reads": s["n_reads"], "assigned_pct": round(100 * s["decisions"].get("assigned", 0) / n, 2)})
        benchmarks["real_read_runs"] = real
        if real:
            charts.append(_chart("real_read_assignment", "Reference real-read runs: % assigned", "hbar",
                                 [r["dataset"] for r in real], [{"name": "assigned %", "values": [r["assigned_pct"] for r in real]}],
                                 "% assigned", "dataset"))
    except HTTPException:
        pass

    return {
        "uploads": {"total_runs": total, "by_status": by_status, "by_marker": by_marker,
                    "total_reads_analysed": sum(r["n_reads"] for r in run_rows), "runs": run_rows},
        "benchmarks": benchmarks,
        "charts": charts,
    }


@router.get("/analytics/compare", tags=["analytics"], summary="Compare two or more finished runs side by side")
def analytics_compare(run_ids: str = Query(..., description="comma-separated run ids, 2-10")):
    ids = [i.strip() for i in run_ids.split(",") if i.strip()]
    if not 2 <= len(ids) <= 10:
        raise HTTPException(400, "give between 2 and 10 run ids")
    runs = [_get_run_or_404(i) for i in ids]
    not_done = [r["id"] for r in runs if r["status"] != "done"]
    if not_done:
        raise HTTPException(409, f"runs not finished yet: {not_done}")

    names = [f"{r['filename']} ({r['id']})" for r in runs]
    table = []
    for r in runs:
        s = r["summary"]
        table.append({"run_id": r["id"], "filename": r["filename"], "marker": r["marker"],
                      "n_reads": s["input"]["n_reads"], "assignment_rate_pct": s["assignment_rate_pct"],
                      "median_identity_assigned": s["identity"]["median_assigned"],
                      "median_coverage": s["coverage"]["median"],
                      "n_distinct_taxa": s["taxa"]["n_distinct_assigned"],
                      "top10_ref_share_pct": s["references"]["top10_share_of_assigned_pct"],
                      **{f"pct_{d['decision']}": d["pct_of_reads"] for d in s["decisions"]}})

    # Taxon x run matrix over assigned reads, from each run's full results.
    counts = []
    for r in runs:
        c: Dict[str, int] = {}
        for row in ra.cached_results(r["results_path"]):
            if row["Confidence_Decision"] == "assigned":
                t = row["Nearest_Organism"] or row["Nearest_Reference_ID"]
                c[t] = c.get(t, 0) + 1
        counts.append(c)
    all_taxa = sorted({t for c in counts for t in c}, key=lambda t: -sum(c.get(t, 0) for c in counts))
    top = all_taxa[:25]
    matrix = [{"taxon": t, **{r["id"]: c.get(t, 0) for r, c in zip(runs, counts)}} for t in top]

    overlap = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = set(counts[i]), set(counts[j])
            overlap.append({"run_a": runs[i]["id"], "run_b": runs[j]["id"], "shared_taxa": len(a & b),
                            "only_a": len(a - b), "only_b": len(b - a),
                            "jaccard": round(len(a & b) / len(a | b), 4) if a | b else None})

    decisions = ra.DECISIONS
    charts = [
        _chart("decision_mix", "Decision mix per run (% of reads)", "stacked_bar", names,
               [{"name": d, "values": [row.get(f"pct_{d}", 0.0) for row in table]} for d in decisions
                if any(row.get(f"pct_{d}") for row in table)], "run", "% of reads"),
        _chart("taxa_by_run", "Top taxa across runs (assigned reads)", "grouped_bar", top,
               [{"name": n, "values": [m[r["id"]] for m in matrix]} for n, r in zip(names, runs)], "taxon", "reads"),
    ]
    markers_used = {r["marker"] for r in runs}
    return {"runs": table, "taxa_matrix": matrix, "taxa_overlap": overlap, "charts": charts,
            "note": None if len(markers_used) == 1 else
            "These runs used different markers, so their taxa come from different reference databases; "
            "compare decision rates, not taxon overlap."}

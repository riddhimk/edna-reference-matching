"""
JSON API over this project's results, plus analysis of uploaded read files.

Two halves:

  * Saved results (this module) -- evaluation metrics, fine-tuning metadata and
    per-read classification outputs that prior experiments wrote, served as-is.

  * Uploads (src/app_routes.py, src/chat.py) -- upload a FASTQ/FASTA file, run
    it through the production pipeline in the background, browse its history,
    results and analytics, and ask the project chatbot about any of it.

Neither half loads a neural model: the production pipeline uses precomputed
reference embeddings and never embeds reads, so the API needs none of
torch/faiss/transformers and starts instantly.

Design notes:

  * Datasets are addressed by a fixed slug, never by a caller-supplied path.
    Paths live in the registries below, so no request can reach outside them.

  * Files are static once written, so reads are cached in-process. Restart the
    server to pick up a re-run experiment.

  * The per-read result files have THREE different schemas, because they were
    produced at different stages of the project (embedding-only, reranked, and
    Tara real-read). Rather than forcing them into one shape and silently
    dropping columns, each dataset reports its own columns and rows are served
    as-is. The generic numeric filter binds to whichever score column that
    dataset actually has.

Usage:
    uvicorn src.api:app --host 127.0.0.1 --port 8000
    then open http://127.0.0.1:8000/docs

Run a single process (no --workers): the upload job queue is in-process.
The chatbot needs ANTHROPIC_API_KEY (or another Anthropic SDK credential);
every other endpoint works without it.
"""
import csv
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

csv.field_size_limit(10 ** 9)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="eDNA-TaxAI Results API",
    version="2.0.0",
    description="Evaluation metrics, fine-tuning runs and per-read results; upload and analyse "
                "new read files; run history, analytics and a project chatbot.",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Registries -- the only paths this API will ever open
# --------------------------------------------------------------------------

PRF_FILES = {
    "bacteria": "data/Precision Recall F1/bacteria_precision_recall_f1.json",
    "fish": "data/Precision Recall F1/fish_precision_recall_f1.json",
    "eukaryota_large": "data/Precision Recall F1/eukaryota_large_precision_recall_f1.json",
}

ARCH_FILES = {
    "bacteria": "data/Finetuned architecture evaluation/results.json",
    "fish": "data/Finetuned architecture evaluation/fish/results.json",
    "eukaryota": "data/Finetuned architecture evaluation/eukaryota/results.json",
}

MODEL_COMPARISON_FILES = {
    "nucleotide_transformer_v2_50m": "data/NT diagnostics/nt_recall_summary.txt",
    "hyenadna_tiny": "data/HyenaDNA diagnostics/recall_summary.txt",
    "gena_lm_bert_base_t2t": "data/GENA-LM diagnostics/recall_summary.txt",
    "dnaberts_frozen_baseline": "data/Stage0 diagnostics/recall_diagnostic_summary.txt",
}

READ_DATASETS = {
    "fish_mitofish_reranked": {
        "path": "data/MitoFish results/read_taxonomy_results_mitofish_16s_reranked.csv",
        "kingdom": "fish", "part": "Part 4", "accession": "ERR16922870",
        "description": "DNABERT top-10 + Smith-Waterman rerank against MitoFish 16S.",
    },
    "fish_mitofish_embedding_only": {
        "path": "data/MitoFish results/read_taxonomy_results_mitofish_16s.csv",
        "kingdom": "fish", "part": "Part 3", "accession": "ERR16922870",
        "description": "Pre-rerank baseline: embedding similarity only.",
    },
    "tara_bacteria_16s_v4v5": {
        "path": "data/SILVA results/tara_16s_v4v5_read_taxonomy_results.csv",
        "kingdom": "bacteria", "part": "Part 10", "accession": "ERR6090675",
        "description": "Real Tara Oceans seawater reads vs SILVA 16S V4-V5.",
    },
    "tara_eukaryota_18s_v9": {
        "path": "data/SILVA results/tara_18s_v9_read_taxonomy_results.csv",
        "kingdom": "eukaryota", "part": "Part 9", "accession": "ERR10928327",
        "description": "Real Tara Pacific seawater reads vs SILVA 18S V9.",
    },
    "silva_16s_v4": {
        "path": "data/SILVA results/read_taxonomy_results_16s_v4.csv",
        "kingdom": "bacteria", "part": "Part 2", "accession": "ERR16922870",
        "description": "Fish reads vs SILVA 16S V4. Superseded: the marker is fish mitochondrial 16S, "
                       "which SILVA does not contain.",
    },
    "silva_18s_v9": {
        "path": "data/SILVA results/read_taxonomy_results_18s_v9.csv",
        "kingdom": "eukaryota", "part": "Part 2", "accession": "ERR16922870",
        "description": "Fish reads vs SILVA 18S V9. Superseded for the same reason.",
    },
    "silva_full_length": {
        "path": "data/SILVA results/read_taxonomy_results.csv",
        "kingdom": "mixed", "part": "Part 1", "accession": "ERR16922870",
        "description": "First run: full-length SILVA references -- the region-mismatch finding.",
    },
}

# Score column varies by schema; first match wins.
SCORE_COLUMNS = ["identity_pct", "Similarity"]


# --------------------------------------------------------------------------
# Loaders (cached -- files are static once written)
# --------------------------------------------------------------------------

def _abs(rel: str) -> str:
    return os.path.join(ROOT, rel)


@lru_cache(maxsize=32)
def load_json(rel: str) -> Any:
    path = _abs(rel)
    if not os.path.exists(path):
        raise HTTPException(404, f"result file not found: {rel}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=32)
def load_csv(rel: str) -> List[Dict[str, str]]:
    path = _abs(rel)
    if not os.path.exists(path):
        raise HTTPException(404, f"result file not found: {rel}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=16)
def load_kv_summary(rel: str) -> Dict[str, Any]:
    """Parses the 'key : value' diagnostic summaries into a dict, coercing
    numbers where possible and keeping the header lines separately."""
    path = _abs(rel)
    if not os.path.exists(path):
        raise HTTPException(404, f"result file not found: {rel}")
    out: Dict[str, Any] = {}
    notes: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if ":" in line and not line.strip().startswith("==="):
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if not k:
                    continue
                try:
                    out[k] = float(v) if "." in v else int(v)
                except ValueError:
                    out[k] = v
            elif line.strip():
                notes.append(line.strip())
    if notes:
        out["_notes"] = notes
    return out


def _score_column(rows: List[Dict[str, str]]) -> Optional[str]:
    if not rows:
        return None
    for c in SCORE_COLUMNS:
        if c in rows[0]:
            return c
    return None


def _as_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Index / health
# --------------------------------------------------------------------------

_FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" fill="none" stroke="#1c5cab" '
            'stroke-width="2.4" stroke-linecap="round"><path d="M8 3c0 7 12 8 12 11s-12 4-12 11"/>'
            '<path d="M20 3c0 7-12 8-12 11s12 4 12 11"/><path d="M10 8h8M9.5 20h9"/></svg>')


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(_FAVICON, media_type="image/svg+xml")


@app.get("/", summary="API index (a browser is sent to the web app)")
def index(request: Request):
    # A person typing the bare address wants the app; scripts and curl
    # (no text/html in Accept) still get this JSON index.
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/app/")
    return {
        "name": "eDNA-TaxAI Results API",
        "loads_neural_model": False,
        "note": "Saved results are read-only. Uploaded files run through the production pipeline "
                "(BM25 -> MMR -> Smith-Waterman) using precomputed reference embeddings.",
        "dashboard": "/app",
        "endpoints": {
            "GET /app": "React results dashboard",
            "GET /health": "service status and which result files are present",
            "GET /kingdoms": "kingdoms and what data exists for each",
            "GET /metrics/prf": "POC-6 precision/recall/F1 (?kingdom, ?n, ?arm)",
            "GET /metrics/prf/comparison": "frozen vs fine-tuned, side by side",
            "GET /metrics/architecture": "7-arm architecture comparison (?kingdom)",
            "GET /models": "fine-tuning metadata for every checkpoint",
            "GET /models/comparison": "DNABERT-S vs the three alternatives",
            "GET /models/{name}": "one checkpoint's metadata",
            "GET /reads/datasets": "per-read result sets available",
            "GET /reads/{dataset}": "per-read rows (?decision, ?min_score, ?limit, ?offset)",
            "GET /reads/{dataset}/summary": "tier counts, score stats, top organisms",
            "GET /markers": "marker presets an upload can be analysed against",
            "POST /uploads/check": "has a file (by SHA-256) already been analysed?",
            "POST /uploads": "upload FASTQ/FASTA (.gz ok) + marker; starts a background run",
            "GET /runs": "history of analysed uploads (?marker, ?status, ?limit, ?offset)",
            "GET /runs/{id}": "status, progress, and full findings + charts when done",
            "GET /runs/{id}/reads": "per-read results (?decision, ?min_identity, ?taxon, ?limit, ?offset)",
            "GET /runs/{id}/download": "per-read results as CSV",
            "DELETE /runs/{id}": "remove a run from history",
            "GET /analytics/overview": "run history + benchmark findings, chart-ready",
            "GET /analytics/compare": "compare 2-10 runs (?run_ids=a,b)",
            "POST /chat/sessions": "start a chatbot conversation (optionally about a run)",
            "POST /chat/sessions/{id}/messages": "ask the project chatbot a question",
            "GET /chat/sessions/{id}": "conversation transcript",
            "GET /docs": "interactive OpenAPI documentation",
        },
    }


@app.get("/health", summary="Service status and data availability")
def health():
    def present(rel):
        return os.path.exists(_abs(rel))

    return {
        "status": "ok",
        "data_root": ROOT,
        "available": {
            "precision_recall_f1": {k: present(v) for k, v in PRF_FILES.items()},
            "architecture_evaluation": {k: present(v) for k, v in ARCH_FILES.items()},
            "model_comparison": {k: present(v) for k, v in MODEL_COMPARISON_FILES.items()},
            "read_datasets": {k: present(v["path"]) for k, v in READ_DATASETS.items()},
            "checkpoints": sorted(
                d for d in os.listdir(_abs("models"))
                if os.path.exists(_abs(f"models/{d}/finetune_metadata.json"))
            ) if os.path.isdir(_abs("models")) else [],
        },
    }


@app.get("/kingdoms", summary="Kingdoms and what data exists for each")
def kingdoms():
    out = {}
    for k in ("fish", "bacteria", "eukaryota", "eukaryota_large"):
        out[k] = {
            "precision_recall_f1": k in PRF_FILES,
            "architecture_evaluation": k in ARCH_FILES,
            "read_datasets": [s for s, m in READ_DATASETS.items() if m["kingdom"] == k],
        }
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@app.get("/metrics/prf", summary="POC-6 precision / recall / F1")
def metrics_prf(
    kingdom: Optional[str] = Query(None, description="bacteria | fish | eukaryota_large"),
    n: Optional[int] = Query(None, description="test-set size: 1000, 2000 or 5000"),
    arm: Optional[str] = Query(None, description="frozen | finetuned"),
):
    targets = [kingdom] if kingdom else list(PRF_FILES)
    for t in targets:
        if t not in PRF_FILES:
            raise HTTPException(404, f"unknown kingdom '{t}'. Options: {list(PRF_FILES)}")

    out = []
    for t in targets:
        d = load_json(PRF_FILES[t])
        rows = d["results"]
        if n is not None:
            rows = [r for r in rows if r["n"] == n]
        if arm is not None:
            rows = [r for r in rows if r["arm"] == arm]
        out.append({
            "kingdom": d["kingdom"],
            "finetuned_model": d["finetuned_model"],
            "identity_threshold": d["identity_threshold"],
            "split": {
                "n_train_refs": d["n_train_refs"],
                "n_holdout_refs": d["n_holdout_refs"],
                "n_fragments": d["n_fragments"],
                "distinct_source_refs": d["distinct_source_refs"],
                "fragments_per_reference": round(d["fragments_per_reference"], 4),
                "fully_independent_samples": d["fragments_per_reference"] <= 1.05,
            },
            "results": rows,
        })
    return {"count": len(out), "kingdoms": out}


@app.get("/metrics/prf/comparison", summary="Frozen vs fine-tuned, side by side")
def metrics_prf_comparison(n: int = Query(5000, description="test-set size to compare at")):
    out = []
    for k, path in PRF_FILES.items():
        d = load_json(path)
        rows = {r["arm"]: r for r in d["results"] if r["n"] == n}
        if "frozen" not in rows or "finetuned" not in rows:
            continue
        fr, ft = rows["frozen"], rows["finetuned"]
        out.append({
            "kingdom": k,
            "n": n,
            "fragments_per_reference": round(d["fragments_per_reference"], 4),
            "fully_independent_samples": d["fragments_per_reference"] <= 1.05,
            "frozen": {"precision": fr["precision"], "recall": fr["recall"], "f1": fr["f1"], "tp": fr["tp"]},
            "finetuned": {"precision": ft["precision"], "recall": ft["recall"], "f1": ft["f1"], "tp": ft["tp"]},
            "delta": {
                "precision": round(ft["precision"] - fr["precision"], 6),
                "recall": round(ft["recall"] - fr["recall"], 6),
                "f1": round(ft["f1"] - fr["f1"], 6),
                "correct_calls": ft["tp"] - fr["tp"],
            },
        })
    if not out:
        raise HTTPException(404, f"no results at n={n}. Try 1000, 2000 or 5000.")
    return {"n": n, "kingdoms": out}


@app.get("/metrics/architecture", summary="7-arm architecture comparison")
def metrics_architecture(kingdom: Optional[str] = Query(None, description="bacteria | fish | eukaryota")):
    targets = [kingdom] if kingdom else list(ARCH_FILES)
    for t in targets:
        if t not in ARCH_FILES:
            raise HTTPException(404, f"unknown kingdom '{t}'. Options: {list(ARCH_FILES)}")
    return {
        "kingdoms": [
            {**{k: v for k, v in load_json(ARCH_FILES[t]).items() if k != "architectures"},
             "architectures": load_json(ARCH_FILES[t])["architectures"]}
            for t in targets
        ]
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

@app.get("/models", summary="Fine-tuning metadata for every checkpoint")
def models():
    base = _abs("models")
    if not os.path.isdir(base):
        return {"count": 0, "models": []}
    out = []
    for name in sorted(os.listdir(base)):
        meta = f"models/{name}/finetune_metadata.json"
        if os.path.exists(_abs(meta)):
            out.append({"name": name, **load_json(meta)})
    return {"count": len(out), "models": out}


@app.get("/models/comparison", summary="DNABERT-S vs the three alternatives")
def models_comparison():
    return {
        "test_set": "1,000 held-out MitoFish fragments, seed 42, identical across all models",
        "models": {k: load_kv_summary(v) for k, v in MODEL_COMPARISON_FILES.items()},
    }


@app.get("/models/{name}", summary="One checkpoint's metadata")
def model_detail(name: str):
    meta = f"models/{name}/finetune_metadata.json"
    if not os.path.exists(_abs(meta)):
        raise HTTPException(404, f"no checkpoint metadata for '{name}'")
    return {"name": name, **load_json(meta)}


# --------------------------------------------------------------------------
# Per-read results
# --------------------------------------------------------------------------

@app.get("/reads/datasets", summary="Per-read result sets available")
def read_datasets():
    out = []
    for slug, meta in READ_DATASETS.items():
        entry = {"dataset": slug, **{k: v for k, v in meta.items() if k != "path"}}
        if os.path.exists(_abs(meta["path"])):
            rows = load_csv(meta["path"])
            entry["n_reads"] = len(rows)
            entry["n_assigned"] = sum(1 for r in rows if r.get("Confidence_Decision") == "assigned")
            entry["columns"] = list(rows[0].keys()) if rows else []
            entry["score_column"] = _score_column(rows)
        else:
            entry["n_reads"] = 0
            entry["available"] = False
        out.append(entry)
    return {"count": len(out), "datasets": out}


@app.get("/reads/{dataset}", summary="Per-read classification rows")
def reads(
    dataset: str,
    decision: Optional[str] = Query(None, description="filter by Confidence_Decision"),
    min_score: Optional[float] = Query(None, description="minimum identity_pct or Similarity"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    if dataset not in READ_DATASETS:
        raise HTTPException(404, f"unknown dataset '{dataset}'. See GET /reads/datasets")
    rows = load_csv(READ_DATASETS[dataset]["path"])
    score_col = _score_column(rows)

    if decision:
        rows = [r for r in rows if r.get("Confidence_Decision") == decision]
    if min_score is not None:
        if not score_col:
            raise HTTPException(400, "this dataset has no numeric score column to filter on")
        rows = [r for r in rows if (_as_float(r.get(score_col)) or -1) >= min_score]

    return {
        "dataset": dataset,
        "score_column": score_col,
        "total_matching": len(rows),
        "limit": limit,
        "offset": offset,
        "reads": rows[offset:offset + limit],
    }


@app.get("/reads/{dataset}/summary", summary="Tier counts, score stats, top organisms", tags=["reads"])
def reads_summary(dataset: str, top: int = Query(10, ge=1, le=100)):
    if dataset not in READ_DATASETS:
        raise HTTPException(404, f"unknown dataset '{dataset}'. See GET /reads/datasets")
    meta = READ_DATASETS[dataset]
    rows = load_csv(meta["path"])
    score_col = _score_column(rows)

    tiers: Dict[str, int] = {}
    for r in rows:
        d = r.get("Confidence_Decision", "unknown")
        tiers[d] = tiers.get(d, 0) + 1

    scores = [s for s in (_as_float(r.get(score_col)) for r in rows) if s is not None] if score_col else []
    scores.sort()

    def pct(p):
        return round(scores[min(int(len(scores) * p), len(scores) - 1)], 4) if scores else None

    label_col = "Nearest_Organism" if rows and "Nearest_Organism" in rows[0] else (
        "Taxonomy" if rows and "Taxonomy" in rows[0] else "Nearest_Reference_ID")
    counts: Dict[str, int] = {}
    for r in rows:
        v = (r.get(label_col) or "").strip() or "(unlabelled)"
        counts[v] = counts.get(v, 0) + 1
    top_labels = sorted(counts.items(), key=lambda kv: -kv[1])[:top]

    return {
        "dataset": dataset,
        "kingdom": meta["kingdom"],
        "part": meta["part"],
        "accession": meta["accession"],
        "n_reads": len(rows),
        "decisions": dict(sorted(tiers.items(), key=lambda kv: -kv[1])),
        "score_column": score_col,
        "score_distribution": {
            "min": round(scores[0], 4) if scores else None,
            "p25": pct(0.25), "median": pct(0.50), "p75": pct(0.75),
            "max": round(scores[-1], 4) if scores else None,
            "mean": round(sum(scores) / len(scores), 4) if scores else None,
        },
        "grouped_by": label_col,
        "distinct_values": len(counts),
        "top": [{"value": v, "reads": c, "pct_of_total": round(100 * c / len(rows), 2)}
                for v, c in top_labels],
    }


# --------------------------------------------------------------------------
# Uploads, run history, analytics and chat
# --------------------------------------------------------------------------

from .app_routes import router as _app_router  # noqa: E402  (needs the functions above defined first)
from .chat import router as _chat_router  # noqa: E402

app.include_router(_app_router)
app.include_router(_chat_router)


# --------------------------------------------------------------------------
# Dashboard (mounted last so it cannot shadow any API route)
#
# The React app is served from this same origin deliberately: it removes the
# CORS problem entirely and means one process serves both data and UI.
# --------------------------------------------------------------------------

WEB_DIR = _abs("web")

if os.path.isdir(WEB_DIR):
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="dashboard")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard_redirect():
        """Convenience alias so /dashboard and /app both work."""
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

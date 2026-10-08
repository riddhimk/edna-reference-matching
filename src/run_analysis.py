"""
Runs an uploaded read file through the production pipeline and summarises it.

The pipeline itself is src/final_pipeline.py, used UNMODIFIED: BM25 -> MMR ->
Smith-Waterman rerank -> tiered decision at the frozen 95% identity bar. This
module only adds what an uploaded file needs around it:

  * Marker presets. A read can only be matched against references cut from
    the same gene region with the same primers (the region-mismatch finding,
    Explanation.md Part 2). Each preset pairs a reference set with the read
    preparation that was validated against it on real data:
        fish_16s_mitofish  -- Part 4 (ERR16922870), raw reads, MitoFish 16S
        bacteria_16s_v4v5  -- Part 10 (ERR6090675), 515F-Y/926R primer trim
        eukaryota_18s_v9   -- Part 9 (ERR10928327), 1389F/1510R primer trim
    The primer regexes and trim logic are imported from those experiments'
    modules, not re-typed, so an upload is prepared exactly as they were.

  * Identical-read collapsing. assign() is deterministic, and amplicon files
    repeat sequences heavily (the 5,000-read fish sample has 1,002 distinct
    sequences; the 18S sample 735), so each distinct sequence is classified
    once and the result shared. The output is identical; the work is not.

  * A process pool. Classification is CPU-bound Smith-Waterman at ~0.19 s per
    read on this machine, so a 5,000-read file is ~15 min serially; with the
    pool and collapsing it takes ~40 s. Workers start from a "forkserver",
    not a plain fork of the API process: a plain fork copies uvicorn's
    listening socket and signal handlers into every worker, so a server that
    died mid-run left workers holding the port and ignoring SIGTERM (seen in
    testing). Each worker loads the reference index itself (~3 s, in parallel)
    and exits on its own if the API process disappears.

No neural model is loaded: reference embeddings were precomputed, and the
production path never embeds reads (MMR uses reference embeddings only).
"""
import csv
import multiprocessing
import os
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

from . import run_tara_16s_v4v5_experiment as tara16s
from . import run_tara_18s_v9_experiment as tara18s
from .final_pipeline import BEST_GUESS_MIN_IDENTITY, BM25_K_MER, FROZEN_IDENTITY_THRESHOLD, PipelineIndex, assign

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Order matters: it is the order decisions are reported in.
DECISIONS = ["assigned", "best_guess", "best_guess_low_confidence", "unclassified", "no_primer"]
# Built from the pipeline's own constants, so the wording can never drift from the rule.
DECISION_NOTES = {
    "assigned": f"identity ≥ {FROZEN_IDENTITY_THRESHOLD:g}% — a confident species-level call",
    "best_guess": f"identity {BEST_GUESS_MIN_IDENTITY:g}–{FROZEN_IDENTITY_THRESHOLD:g}% — a plausible lead, "
                  f"below the calibrated confidence bar",
    "best_guess_low_confidence": f"identity < {BEST_GUESS_MIN_IDENTITY:g}% — nearest known relative only, weak match",
    "unclassified": f"no reference shares any {BM25_K_MER}-mer with the read",
    "no_primer": "the marker's primer was not found at the read start, so the read was excluded "
                 "(it may come from a different amplicon)",
}

# A run is flagged when its top 10 references take at least this share of the
# assigned reads -- the pattern the project calls hubness (Explanation.md Part 2
# s.12, Part 3 s.22). A flag to look closer, not a verdict.
HUBNESS_FLAG_PCT = 80.0
IDENTITY_BIN_WIDTH = 5

RESULT_COLUMNS = ["Read_ID", "Nearest_Reference_ID", "Nearest_Organism", "Taxonomy", "identity_pct",
                  "coverage_pct", "bm25_score", "Confidence_Decision"]


@dataclass(frozen=True)
class Marker:
    key: str
    label: str
    kingdom: str
    description: str
    # The earlier real-read run this preset reproduces: its project part, the
    # sequencing accession, and its saved per-read results (a slug in
    # api.READ_DATASETS) -- figures about it are computed from that file.
    validated_part: str
    validated_accession: str
    validated_dataset: str
    ref_table: str
    ref_embeddings: str
    ref_ids: str
    # Returns the trimmed, correctly-oriented read, or None when no primer is found.
    # None for markers whose reads are classified as uploaded.
    trim: Optional[Callable[[str], Optional[str]]] = None
    primers: Optional[str] = None
    # MitoFish carries no species names; they are looked up from NCBI by accession.
    species_from_ncbi: bool = False


MARKERS: Dict[str, Marker] = {m.key: m for m in [
    Marker(
        key="fish_16s_mitofish", label="Fish — mitochondrial 16S (MitoFish)", kingdom="fish",
        description="Fish eDNA amplified at mitochondrial 16S, matched against MitoFish 16S references.",
        validated_part="Part 4", validated_accession="ERR16922870", validated_dataset="fish_mitofish_reranked",
        ref_table="data/MitoFish results/mitofish_16s_reference_table.csv",
        ref_embeddings="data/MitoFish results/mitofish_16s_ref_embeddings.npy",
        ref_ids="data/MitoFish results/mitofish_16s_ref_ids.csv",
        species_from_ncbi=True,
    ),
    Marker(
        key="bacteria_16s_v4v5", label="Bacteria/archaea — 16S V4-V5 (SILVA)", kingdom="bacteria",
        description="Prokaryote 16S V4-V5 amplicons (515F-Y / 926R), matched against SILVA references "
                    "cut with the same primers.",
        validated_part="Part 10", validated_accession="ERR6090675", validated_dataset="tara_bacteria_16s_v4v5",
        ref_table="data/SILVA results/silva_16s_v4v5_tara_reference_table.csv",
        ref_embeddings="data/SILVA results/silva_16s_v4v5_tara_ref_embeddings.npy",
        ref_ids="data/SILVA results/silva_16s_v4v5_tara_ref_ids.csv",
        trim=tara16s.orient_and_trim, primers="515F-Y / 926R",
    ),
    Marker(
        key="eukaryota_18s_v9", label="Eukaryotic plankton — 18S V9 (SILVA)", kingdom="eukaryota",
        description="Eukaryote 18S V9 amplicons (1389F / 1510R), matched against SILVA references "
                    "cut with the same primers.",
        validated_part="Part 9", validated_accession="ERR10928327", validated_dataset="tara_eukaryota_18s_v9",
        ref_table="data/SILVA results/silva_18s_v9_tara_reference_table.csv",
        ref_embeddings="data/SILVA results/silva_18s_v9_tara_ref_embeddings.npy",
        ref_ids="data/SILVA results/silva_18s_v9_tara_ref_ids.csv",
        trim=tara18s.orient_and_trim, primers="1389F / 1510R",
    ),
]}


def _abs(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def marker_available(m: Marker) -> bool:
    return all(os.path.exists(_abs(p)) for p in (m.ref_table, m.ref_embeddings, m.ref_ids))


@lru_cache(maxsize=None)
def count_references(marker_key: str) -> Optional[int]:
    path = _abs(MARKERS[marker_key].ref_ids)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)          # minus the header row


def validation_stats(m: Marker) -> Optional[dict]:
    """Assigned / total from the saved per-read results of the run this
    preset reproduces -- read from the file, never typed in."""
    from . import api   # api imports this module (via app_routes); import lazily
    meta = api.READ_DATASETS.get(m.validated_dataset)
    if not meta or not os.path.exists(_abs(meta["path"])):
        return None
    rows = api.load_csv(meta["path"])
    assigned = sum(1 for r in rows if r.get("Confidence_Decision") == "assigned")
    return {"dataset": m.validated_dataset, "n_reads": len(rows), "n_assigned": assigned,
            "assigned_pct": round(100 * assigned / len(rows), 2) if rows else None}


def describe_markers() -> List[dict]:
    return [{"key": m.key, "label": m.label, "kingdom": m.kingdom, "description": m.description,
             "validated_part": m.validated_part, "validated_accession": m.validated_accession,
             "validation": validation_stats(m), "n_references": count_references(m.key),
             "primers": m.primers, "primer_trimming": m.trim is not None,
             "identity_threshold": FROZEN_IDENTITY_THRESHOLD, "best_guess_min_identity": BEST_GUESS_MIN_IDENTITY,
             "available": marker_available(m)} for m in MARKERS.values()]


@lru_cache(maxsize=None)
def load_index(marker_key: str) -> PipelineIndex:
    m = MARKERS[marker_key]
    return PipelineIndex.build(_abs(m.ref_table), _abs(m.ref_embeddings), _abs(m.ref_ids))


@lru_cache(maxsize=None)
def load_taxonomy(marker_key: str) -> Dict[str, Tuple[str, str]]:
    """{Reference_ID: (organism_label, full_taxonomy)} from the reference table."""
    out = {}
    with open(_abs(MARKERS[marker_key].ref_table), newline="") as f:
        for row in csv.DictReader(f):
            tax = row.get("Taxonomy_Full", "") or ""
            label = (row.get("Organism_Label") or "").strip() or (tax.split(";")[-1].strip() if tax else "")
            out[row["Reference_ID"]] = (label, tax)
    return out


# ------------------------------------------------------- marker detection ---
#
# Which reference set fits an upload is decided from the reads themselves: the
# share of each read's 12-mers that occur anywhere in a reference set. Reads
# from a set's gene region share most of their 12-mers with it; reads from
# another region share almost none (the region-mismatch finding, measured).
#
# Calibration, measured on the three validated real-read files plus an
# off-target 23S FASTA (mean share over 400 distinct reads, best orientation):
#     reads \ refs          fish 16S   bact 16S   euk 18S
#     ERR16922870 (fish)     0.339      0.013      0.001
#     Tara 16S V4-V5         0.010      0.827      0.003
#     Tara 18S V9            0.012      0.008      0.411
#     23S rRNA FASTA         0.025      0.024      0.006
# Every matching pair scored >= 0.34, every non-matching pair <= 0.025. The
# thresholds below sit inside that gap with room on both sides.
DETECT_K = 12
DETECT_SAMPLE = 400
DETECT_MIN_SCORE = 0.10      # best set must share at least this much...
DETECT_MIN_RATIO = 4.0       # ...and this many times more than the runner-up
DETECT_PRIMER_MIN = 0.20     # below this primer rate, reads are taken as already trimmed

_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def _kmers(seq: str, k: int = DETECT_K) -> set:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


@lru_cache(maxsize=None)
def reference_kmers(marker_key: str) -> frozenset:
    ks = set()
    with open(_abs(MARKERS[marker_key].ref_table), newline="") as f:
        for row in csv.DictReader(f):
            ks |= _kmers(row["Sequence"].upper())
    return frozenset(ks)


def _share(seq: str, ks: frozenset) -> float:
    km = _kmers(seq)
    return len(km & ks) / len(km) if km else 0.0


def orient(seq: str, ks: frozenset) -> str:
    """The read in whichever orientation shares more k-mers with the references."""
    rc = revcomp(seq)
    return rc if _share(rc, ks) > _share(seq, ks) else seq


def detect_marker(sequences: List[str], seed: int = 0) -> dict:
    """Scores every available reference set against a sample of the reads.
    Returns the best set, how the reads should be prepared for it, whether the
    call is confident, and the evidence for every set."""
    uniq = list(dict.fromkeys(s for s in sequences if len(s) >= DETECT_K))
    sample = random.Random(seed).sample(uniq, min(DETECT_SAMPLE, len(uniq)))
    evidence = []
    for key, m in MARKERS.items():
        if not marker_available(m):
            continue
        ks = reference_kmers(key)
        primer_frac, mode, reads = None, "as_uploaded", sample
        if m.trim is not None and sample:
            trimmed = [m.trim(s) for s in sample]
            primer_frac = sum(t is not None for t in trimmed) / len(sample)
            if primer_frac >= DETECT_PRIMER_MIN:
                mode, reads = "primer_trim", [t for t in trimmed if t]
        scores = [_share(r, ks) if mode == "primer_trim" else max(_share(r, ks), _share(revcomp(r), ks))
                  for r in reads if len(r) >= DETECT_K]
        evidence.append({"marker": key, "label": m.label, "score": round(statistics.mean(scores), 4) if scores else 0.0,
                         "primer_fraction": None if primer_frac is None else round(primer_frac, 3), "mode": mode})
    evidence.sort(key=lambda e: -e["score"])
    if not evidence:
        raise RuntimeError("no reference set is available on this server")
    best = evidence[0]
    runner = evidence[1]["score"] if len(evidence) > 1 else 0.0
    confident = best["score"] >= DETECT_MIN_SCORE and best["score"] >= DETECT_MIN_RATIO * max(runner, 1e-9)
    if not sample:
        reason = f"no read is at least {DETECT_K} bases long, so the gene region could not be identified"
        confident = False
    elif confident:
        reason = (f"{best['score']:.1%} of the reads' {DETECT_K}-mers occur in {best['label']}, "
                  f"vs {runner:.1%} for the next best")
    else:
        reason = (f"the closest, {best['label']}, shares only "
                  f"{best['score']:.1%} of their {DETECT_K}-mers (a match is >= {DETECT_MIN_SCORE:.0%}"
                  + (f" and {DETECT_MIN_RATIO:g}x the runner-up" if len(evidence) > 1 else "") + ")")
    return {"marker": best["marker"], "mode": best["mode"], "confident": confident, "reason": reason,
            "sample_size": len(sample), "k": DETECT_K, "evidence": evidence}


# --------------------------------------------------------------- workers ---

_WORKER_MARKER: Optional[str] = None   # set in each pool worker by _init_worker


def _exit_when_server_dies(server_pid: int) -> None:
    # Workers are children of the forkserver, not of the API process, so they
    # are not told when the API process dies (e.g. kill -9). Without this they
    # sit idle forever after a crash -- seen in testing.
    while True:
        time.sleep(2)
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            os._exit(1)


def _init_worker(marker_key: str, marker: "Marker", server_pid: int) -> None:
    global _WORKER_MARKER
    threading.Thread(target=_exit_when_server_dies, args=(server_pid,), daemon=True).start()
    MARKERS[marker_key] = marker        # a worker starts fresh, so carry the preset over (tests add their own)
    _WORKER_MARKER = marker_key
    load_index(marker_key)


def _classify_chunk(seqs: List[str]) -> List[dict]:
    index = load_index(_WORKER_MARKER)
    return [assign(s, index) for s in seqs]


def default_workers() -> int:
    # All cores but two (left for the API and the OS); EDNA_WORKERS overrides.
    return int(os.environ.get("EDNA_WORKERS", max(1, (os.cpu_count() or 2) - 2)))


def classify_unique(seqs: List[str], marker_key: str, workers: int,
                    on_progress: Callable[[int], None], chunk: int = 20) -> Dict[str, dict]:
    """Classifies each distinct sequence once. Returns {sequence: assign() result}."""
    unique = list(dict.fromkeys(seqs))
    out: Dict[str, dict] = {}
    chunks = [unique[i:i + chunk] for i in range(0, len(unique), chunk)]

    # Small inputs aren't worth the ~3 s of workers loading the index.
    # Daemonic processes (e.g. inside ZeroGPU or web worker processes) cannot spawn child processes.
    is_daemon = getattr(multiprocessing.current_process(), "daemon", False)
    if workers <= 1 or len(unique) < 200 or is_daemon:
        index = load_index(marker_key)
        for c in chunks:
            for s in c:
                out[s] = assign(s, index)
            on_progress(len(out))
        return out

    try:
        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(marker_key, MARKERS[marker_key], os.getpid())) as pool:
            futures = {pool.submit(_classify_chunk, c): c for c in chunks}
            for fut in as_completed(futures):
                for s, r in zip(futures[fut], fut.result()):
                    out[s] = r
                on_progress(len(out))
        return out
    except (AssertionError, ValueError, RuntimeError):
        index = load_index(marker_key)
        for c in chunks:
            for s in c:
                out[s] = assign(s, index)
            on_progress(len(out))
        return out


# -------------------------------------------------------------- the run ---

@dataclass
class RunOutput:
    rows: List[dict]
    summary: dict
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def run_marker(marker_key: str, read_ids: List[str], sequences: List[str],
               on_progress: Callable[[int, int], None] = lambda done, total: None,
               workers: Optional[int] = None, resolve_species: bool = True,
               mode: Optional[str] = None, detection: Optional[dict] = None) -> RunOutput:
    """mode: "primer_trim" or "as_uploaded"; None = the preset's validated default
    (trim when it has primers). "as_uploaded" for a primer preset means the
    reads were already trimmed: each is oriented against the references instead."""
    m = MARKERS[marker_key]
    t0 = time.time()

    # 1. Read preparation, exactly as the validated experiment did it.
    primer_counts = None
    prepared: List[Optional[str]] = list(sequences)
    if m.trim is not None and mode != "as_uploaded":
        prepared = [m.trim(s) for s in sequences]
        primer_counts = {"primer_found": sum(p is not None for p in prepared),
                         "no_primer": sum(p is None for p in prepared)}
        preparation = f"primer trim ({m.primers})"
    elif m.trim is not None:
        ks = reference_kmers(marker_key)
        prepared = [orient(s, ks) for s in sequences]
        preparation = f"reads used as uploaded, oriented against the references ({m.primers} primers not found: already trimmed)"
    else:
        preparation = "reads used as uploaded"
    usable = [p for p in prepared if p is not None]      # an empty read is kept: it simply matches nothing

    # 2. Classify each distinct usable sequence once.
    n_unique = len(set(usable))
    on_progress(0, n_unique)
    by_seq = classify_unique(usable, marker_key, workers or default_workers(),
                             lambda done: on_progress(done, n_unique))
    n_refs = len(load_index(marker_key).ref_ids)

    # 3. Names for the nearest references.
    taxonomy = load_taxonomy(marker_key)
    names: Dict[str, str] = {}
    refs = sorted({r["Nearest_Reference_ID"] for r in by_seq.values() if r["Nearest_Reference_ID"]})
    if m.species_from_ncbi and resolve_species and refs:
        from .species_lookup import fetch_organism_names
        names = fetch_organism_names(refs, cache_path=_abs("data/species_name_cache.csv"))

    rows = []
    for rid, seq in zip(read_ids, prepared):
        if seq is None:
            rows.append({"Read_ID": rid, "Nearest_Reference_ID": "", "Nearest_Organism": "", "Taxonomy": "",
                         "identity_pct": "", "coverage_pct": "", "bm25_score": "", "Confidence_Decision": "no_primer"})
            continue
        r = by_seq[seq]
        ref = r["Nearest_Reference_ID"]
        label, tax = taxonomy.get(ref, ("", "")) if ref else ("", "")
        rows.append({"Read_ID": rid, "Nearest_Reference_ID": ref,
                     "Nearest_Organism": names.get(ref) or label, "Taxonomy": tax,
                     "identity_pct": round(r["identity_pct"], 2), "coverage_pct": round(r["coverage_pct"], 2),
                     "bm25_score": round(r["bm25_score"], 4), "Confidence_Decision": r["Confidence_Decision"]})

    seconds = round(time.time() - t0, 1)
    summary = summarise(rows, sequences, m, n_unique, primer_counts, n_refs, seconds)
    summary["input"]["preparation"] = preparation
    if detection is not None:
        summary["marker_detection"] = detection
    return RunOutput(rows=rows, summary=summary, seconds=seconds)


def write_results_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_results_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cached_results(path: str) -> List[dict]:
    """read_results_csv, cached per file version: a large run's results are
    parsed once, not on every page of the read table. Keyed on mtime, so a
    rewritten file is reloaded."""
    return _cached_results(path, os.path.getmtime(path))


@lru_cache(maxsize=4)
def _cached_results(path: str, mtime: float) -> List[dict]:
    return read_results_csv(path)


# ------------------------------------------------------------- summary ---

def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 2) if d else 0.0


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def identity_histogram(identities: List[float], width: int = IDENTITY_BIN_WIDTH) -> List[dict]:
    bins = Counter(min(int(v // width) * width, 100 - width) for v in identities)
    return [{"bin_start": b, "bin_end": b + width, "reads": bins.get(b, 0)} for b in range(0, 100, width)]


def summarise(rows: List[dict], raw_sequences: List[str], m: Marker, n_unique: int,
              primer_counts: Optional[dict], n_refs: int, seconds: float) -> dict:
    """Everything the results dashboard shows for one run. Percentages are of
    ALL reads in the file unless a field says otherwise."""
    n = len(rows)
    lengths = [len(s) for s in raw_sequences]
    dec = Counter(r["Confidence_Decision"] for r in rows)
    classified = [r for r in rows if r["Confidence_Decision"] not in ("no_primer",)]
    assigned = [r for r in rows if r["Confidence_Decision"] == "assigned"]
    ident_all = [v for v in (_num(r["identity_pct"]) for r in classified) if v is not None]
    ident_asg = [v for v in (_num(r["identity_pct"]) for r in assigned) if v is not None]
    cov_all = [v for v in (_num(r["coverage_pct"]) for r in classified) if v is not None]

    def top_taxa(subset: List[dict], k: int = 15) -> List[dict]:
        groups = defaultdict(list)
        for r in subset:
            groups[r["Nearest_Organism"] or r["Nearest_Reference_ID"] or "(no match)"].append(_num(r["identity_pct"]) or 0.0)
        ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:k]
        return [{"taxon": t, "reads": len(v), "pct_of_subset": _pct(len(v), len(subset)),
                 "mean_identity": round(statistics.mean(v), 2)} for t, v in ranked]

    def lineage(subset: List[dict], level: int) -> List[dict]:
        c = Counter()
        for r in subset:
            parts = [p for p in (r["Taxonomy"] or "").split(";") if p]
            c[parts[level] if len(parts) > level else "(not resolved at this level)"] += 1
        return [{"taxon": t, "reads": v, "pct_of_assigned": _pct(v, len(subset))} for t, v in c.most_common(15)]

    ref_hits = Counter(r["Nearest_Reference_ID"] for r in assigned)
    top10 = sum(c for _, c in ref_hits.most_common(10))

    return {
        "marker": m.key,
        "marker_label": m.label,
        "reference": {"n_references": n_refs, "table": m.ref_table},
        "identity_threshold": FROZEN_IDENTITY_THRESHOLD,
        "best_guess_min_identity": BEST_GUESS_MIN_IDENTITY,
        "input": {
            "n_reads": n,
            "n_unique_sequences_classified": n_unique,
            "read_length": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths),
                            "mean": round(statistics.mean(lengths), 1)} if lengths else None,
            "primers": m.primers,
            "primer_detection": primer_counts,
        },
        "decisions": [{"decision": d, "reads": dec.get(d, 0), "pct_of_reads": _pct(dec.get(d, 0), n),
                       "meaning": DECISION_NOTES[d]} for d in DECISIONS if dec.get(d, 0) or d != "no_primer"],
        "assignment_rate_pct": _pct(len(assigned), n),
        # Of the reads that reached the classifier (primer found). This is the
        # rate the project's reports quote, e.g. Part 10's 66.9% = 3,189/4,767.
        "assignment_rate_of_analysed_pct": _pct(len(assigned), len(classified)),
        "identity": {
            "median_all_classified": round(statistics.median(ident_all), 2) if ident_all else None,
            "median_assigned": round(statistics.median(ident_asg), 2) if ident_asg else None,
            "histogram_bin_width": IDENTITY_BIN_WIDTH,
            "histogram": identity_histogram(ident_all),
        },
        "coverage": {"median": round(statistics.median(cov_all), 2) if cov_all else None},
        "taxa": {
            "n_distinct_assigned": len({r["Nearest_Organism"] or r["Nearest_Reference_ID"] for r in assigned}),
            "top_assigned": top_taxa(assigned),
            "top_nearest_all_classified": top_taxa(classified),
            # Level 1 is phylum for SILVA bacteria; eukaryote SILVA lineages are
            # clade-based, so level 1 there is a supergroup, not a formal rank.
            "lineage_level_1_assigned": lineage(assigned, 1) if any(r["Taxonomy"] for r in assigned) else [],
        },
        "references": {
            "distinct_hit_by_assigned": len(ref_hits),
            # The project's known failure mode is "hubness": a few references
            # absorbing matches from many unrelated reads (Explanation.md Part 2
            # s.12, Part 3 s.22). A very high share here is a prompt to check.
            "top10_share_of_assigned_pct": _pct(top10, len(assigned)),
            "hubness_flag_pct": HUBNESS_FLAG_PCT,
            "concentrated": bool(assigned) and _pct(top10, len(assigned)) >= HUBNESS_FLAG_PCT,
        },
        "seconds": seconds,
    }

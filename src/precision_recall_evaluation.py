"""
Precision / Recall / F1 for the FULL production pipeline, frozen vs fine-tuned
embeddings, at three test-set sizes (1,000 / 2,000 / 5,000 fragments) per kingdom.

Every earlier evaluation in this project reported Recall@K, which measures only
whether the true match appeared somewhere in a top-K list. That is a retrieval
metric, not a decision metric: it says nothing about whether the pipeline
actually COMMITTED to the right answer, or wrongly committed to a bad one. This
module scores the thing the pipeline actually outputs -- its tiered confidence
decision -- as a binary classifier.

Framing. For each test fragment the pipeline either commits ("assigned", >=95%
identity to its chosen reference) or abstains (best_guess / low-confidence /
unclassified). A commit is correct when the chosen reference really is >=95%
identical to the fragment's true source sequence, checked with the same frozen
Smith-Waterman ground truth (nucleotide_rerank.align_pair) used everywhere else.

    TP = committed and correct
    FP = committed and wrong
    FN = abstained
    precision = TP / (TP + FP)      of the calls it made, how many were right
    recall    = TP / (TP + FN)      of all fragments, how many it got right
    F1        = harmonic mean

The recall denominator is ALL fragments, including any whose true source has no
>=95% relative anywhere in the reference set. Those are genuinely unanswerable,
so recall here is a floor, not a ceiling -- stated rather than silently adjusted,
because computing true findability would need an alignment of every fragment
against every training reference (5,000 x 20,000 for eukaryota_large).

Test-set sizes are NESTED prefixes of one 5,000-fragment run: the 1,000-sample
row is the first 1,000 of the same fragments, so the three sizes are directly
comparable and cost one pass instead of three. Fragments are cut from held-out
references only, using the identical seed-42 partition each model was trained
against, so nothing here was seen in training.

Where a kingdom has fewer held-out references than requested fragments, multiple
DIFFERENT random windows are cut from the same reference and the
fragments-per-reference ratio is reported. Those fragments are not statistically
independent; eukaryota_large exists precisely so that at least one kingdom needs
no such re-sampling.

Usage:
    python -m src.precision_recall_evaluation <kingdom> [finetuned_model_dir]
    kingdom in {fish, bacteria, eukaryota, eukaryota_large}
"""
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from .bm25_retrieval import BM25Index
from .mmr import mmr_select
from .nucleotide_rerank import rerank_candidates, align_pair
from .faiss_index import l2_normalize
from .final_pipeline import (
    BM25_K_MER, MMR_LAMBDA, MMR_POOL_SIZE, TOP_K, FROZEN_IDENTITY_THRESHOLD,
)

torch.set_num_threads(os.cpu_count() or 8)

SEED = 42
HOLDOUT_FRAC = 0.2                     # identical to every prior split in this project
REAL_READ_LENGTHS = list(range(118, 141))
SAMPLE_SIZES = [1000, 2000, 5000]
EMBED_BATCH = 32
OUT_DIR = "data/Precision Recall F1"

DEFAULT_FINETUNED_MODEL_DIR = {
    "bacteria": "models/dnaberts_finetuned_bacteria_v2",
    "fish": "models/dnaberts_finetuned_fish_v2",
    "eukaryota": "models/dnaberts_finetuned_eukaryota_v2",
    "eukaryota_large": "models/dnaberts_finetuned_eukaryota_large",
}


def _load_reference_table(path):
    ids, seqs = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])
    return ids, seqs


def build_split_and_fragments(kingdom: str, n_fragments: int):
    """Rebuilds the EXACT seed-42 train/holdout partition each model was trained
    against, then cuts n_fragments from the held-out side only.

    Replicates the partition arithmetic of stage0_diagnostics /
    other_kingdom_diagnostics rather than calling them, because both cap the
    fragment count at their module-level N_TEST=1000 and cut at most one
    fragment per reference. The PARTITION itself is identical -- same seed, same
    RandomState.permutation, same holdout_frac -- which is what guarantees the
    held-out references were excluded from training.
    """
    if kingdom == "fish":
        from .stage0_diagnostics import REF_TABLE_CSV, REF_EMB_NPY, REF_IDS_CSV
        ref_table_csv, ref_emb_npy, ref_ids_csv = REF_TABLE_CSV, REF_EMB_NPY, REF_IDS_CSV
    else:
        from .evaluate_finetuned import REF_TABLES
        ref_table_csv, ref_emb_npy, ref_ids_csv = REF_TABLES[kingdom]

    ids, seqs = _load_reference_table(ref_table_csv)
    all_emb = np.load(ref_emb_npy).astype("float32")
    with open(ref_ids_csv) as f:
        emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
    assert emb_ids == ids, "ref_ids.csv order does not match reference table order"

    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)
    n = len(ids)
    perm = np_rng.permutation(n)
    n_holdout = int(n * HOLDOUT_FRAC)
    holdout_idx = list(perm[:n_holdout])
    train_idx = perm[n_holdout:]

    train_ids = [ids[i] for i in train_idx]
    train_seqs = [seqs[i] for i in train_idx]
    frozen_train_emb = l2_normalize(all_emb[train_idx])

    # Only references strictly longer than the fragment can yield one.
    usable = [i for i in holdout_idx if len(seqs[i]) > min(REAL_READ_LENGTHS)]
    synth_seqs, synth_true_seq, synth_source_ref = [], [], []
    pos = 0
    while len(synth_seqs) < n_fragments and usable:
        i = usable[pos % len(usable)]
        pos += 1
        seq = seqs[i]
        L = rng.choice(REAL_READ_LENGTHS)
        if len(seq) <= L:
            continue
        start = rng.randint(0, len(seq) - L)
        synth_seqs.append(seq[start:start + L])
        synth_true_seq.append(seq)
        synth_source_ref.append(ids[i])
        if pos > n_fragments * 50:      # pathological guard, never hit in practice
            break

    return {
        "train_ids": train_ids, "train_seqs": train_seqs,
        "frozen_train_emb": frozen_train_emb,
        "synth_seqs": synth_seqs, "synth_true_seq": synth_true_seq,
        "synth_source_ref": synth_source_ref,
        "n_holdout_refs": len(usable),
    }


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def load_finetuned_embed_fn(model_dir: str):
    import multimolecule  # noqa: F401
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()

    def embed_batch(seqs):
        enc = tok(seqs, padding=True, truncation=True, max_length=300, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.numpy().astype("float32")

    return embed_batch


def _embed_all(embed_fn, seqs, label):
    vecs = []
    t0 = time.time()
    for i in range(0, len(seqs), EMBED_BATCH):
        vecs.append(embed_fn(seqs[i:i + EMBED_BATCH]))
        if (i // EMBED_BATCH) % 60 == 0:
            print(f"  [{label}] {i + len(vecs[-1])}/{len(seqs)}  ({time.time() - t0:.0f}s)", flush=True)
    return np.vstack(vecs).astype("float32")


def run_pipeline_arm(label, ref_emb, bm25_pool, train_ids, train_seqs,
                     synth_seqs, synth_true_seq):
    """Runs the production arm (BM25 -> MMR -> frozen reranker -> 95% decision)
    and returns per-fragment (committed, correct) outcomes."""
    n = len(synth_seqs)
    committed, correct = [], []
    train_id_to_seq = dict(zip(train_ids, train_seqs))
    t0 = time.time()

    for i in range(n):
        pool = bm25_pool[i]
        doc_idxs = [d for d, _s in pool]
        scores = [s for _d, s in pool]
        if not doc_idxs:
            committed.append(False)
            correct.append(False)
            continue

        chosen = mmr_select(doc_idxs, scores, ref_emb[doc_idxs],
                            lam=MMR_LAMBDA, k_results=TOP_K)
        score_map = {d: s for d, s in pool}
        candidates = [(train_ids[idx], train_seqs[idx], score_map.get(idx, 0.0))
                      for idx in chosen]
        best = rerank_candidates(synth_seqs[i], candidates)[0]

        is_committed = best["identity_pct"] >= FROZEN_IDENTITY_THRESHOLD
        chosen_seq = train_id_to_seq.get(best["Reference_ID"])
        is_correct = bool(chosen_seq is not None and _is_hit(chosen_seq, synth_true_seq[i]))
        committed.append(is_committed)
        correct.append(is_correct)

        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i + 1}/{n}  ({time.time() - t0:.0f}s)", flush=True)

    return np.array(committed), np.array(correct)


def metrics_at(committed, correct, n):
    """Precision / recall / F1 over the first n fragments (nested prefix)."""
    c, k = committed[:n], correct[:n]
    tp = int((c & k).sum())
    fp = int((c & ~k).sum())
    fn = int((~c).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "committed": int(c.sum()), "committed_pct": float(c.mean()),
    }


def main(kingdom: str, finetuned_model_dir: str = None):
    finetuned_model_dir = finetuned_model_dir or DEFAULT_FINETUNED_MODEL_DIR[kingdom]
    os.makedirs(OUT_DIR, exist_ok=True)
    max_n = max(SAMPLE_SIZES)

    print(f"=== Precision / Recall / F1 -- {kingdom} ===", flush=True)
    print(f"frozen vs fine-tuned ({finetuned_model_dir}), production arm BM25 -> MMR -> reranker\n", flush=True)

    split = build_split_and_fragments(kingdom, max_n)
    train_ids, train_seqs = split["train_ids"], split["train_seqs"]
    synth_seqs, synth_true_seq = split["synth_seqs"], split["synth_true_seq"]
    n = len(synth_seqs)
    n_holdout_refs = split["n_holdout_refs"]
    frags_per_ref = n / n_holdout_refs if n_holdout_refs else float("nan")
    distinct_sources = len(set(split["synth_source_ref"]))

    print(f"train refs        : {len(train_ids):,}")
    print(f"held-out refs     : {n_holdout_refs:,}")
    print(f"fragments built   : {n:,}  (from {distinct_sources:,} distinct refs, "
          f"{frags_per_ref:.2f} per ref)")
    if frags_per_ref > 1.05:
        print(f"  NOTE: >1 fragment per reference -- these are NOT fully independent samples.")
    print(flush=True)

    # --- BM25 is embedding-independent: build once, share across both arms ---
    print("Building BM25 index...", flush=True)
    bm25 = BM25Index(k=BM25_K_MER)
    bm25.build(train_ids, train_seqs)
    bm25_pool = [bm25.top_k(s, k_results=MMR_POOL_SIZE) for s in synth_seqs]

    # --- frozen arm (reference embeddings already on disk) ---
    frozen_ref_emb = split["frozen_train_emb"]
    print("\n--- FROZEN arm ---", flush=True)
    fr_committed, fr_correct = run_pipeline_arm(
        "frozen", frozen_ref_emb, bm25_pool, train_ids, train_seqs, synth_seqs, synth_true_seq)

    # --- fine-tuned arm ---
    print("\nEmbedding references with FINE-TUNED model...", flush=True)
    ft_embed = load_finetuned_embed_fn(finetuned_model_dir)
    ft_ref_emb = l2_normalize(_embed_all(ft_embed, train_seqs, "ft-refs"))
    print("\n--- FINE-TUNED arm ---", flush=True)
    ft_committed, ft_correct = run_pipeline_arm(
        "finetuned", ft_ref_emb, bm25_pool, train_ids, train_seqs, synth_seqs, synth_true_seq)

    # --- metrics at each nested sample size ---
    rows = []
    for size in SAMPLE_SIZES:
        if size > n:
            print(f"(skipping n={size}: only {n} fragments available)")
            continue
        for arm, com, cor in (("frozen", fr_committed, fr_correct),
                              ("finetuned", ft_committed, ft_correct)):
            m = metrics_at(com, cor, size)
            m.update({"kingdom": kingdom, "arm": arm})
            rows.append(m)

    out_json = f"{OUT_DIR}/{kingdom}_precision_recall_f1.json"
    with open(out_json, "w") as f:
        json.dump({
            "kingdom": kingdom,
            "finetuned_model": finetuned_model_dir,
            "identity_threshold": FROZEN_IDENTITY_THRESHOLD,
            "n_train_refs": len(train_ids),
            "n_holdout_refs": n_holdout_refs,
            "n_fragments": n,
            "distinct_source_refs": distinct_sources,
            "fragments_per_reference": frags_per_ref,
            "results": rows,
        }, f, indent=2)
    with open(f"{OUT_DIR}/{kingdom}_precision_recall_f1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== Summary ({kingdom}) ===")
    print(f"{'n':>6s} {'arm':>10s} {'precision':>10s} {'recall':>8s} {'F1':>8s} "
          f"{'TP':>6s} {'FP':>5s} {'FN':>6s}")
    for r in rows:
        print(f"{r['n']:6d} {r['arm']:>10s} {r['precision']:10.4f} {r['recall']:8.4f} "
              f"{r['f1']:8.4f} {r['tp']:6d} {r['fp']:5d} {r['fn']:6d}")
    print(f"\nOutputs written to: {OUT_DIR}")
    return rows


if __name__ == "__main__":
    _kingdom = sys.argv[1] if len(sys.argv) > 1 else "bacteria"
    _model = sys.argv[2] if len(sys.argv) > 2 else None
    main(_kingdom, _model)

"""
Validates the production architecture (BM25 -> MMR -> nucleotide reranker
-> tiered confidence decision, src/final_pipeline.py) on kingdoms other
than fish -- professor's "other species than fish" next-steps item.

Data source: reuses the Bacteria/Archaea (16S V4) and Eukaryota (18S V9)
SILVA-derived reference tables + DNABERT embeddings Part 2 already built
(data/SILVA results/silva_16s_v4_sample.csv, silva_18s_v9_sample.csv),
rather than sourcing brand-new datasets. These ARE genuinely different
kingdoms from fish, and were already verified via primer-based in-silico
PCR against the real 515F/806R and 1391F/EukBr primer pairs (Explanation.md
Part 2, Sections 9-10) -- not arbitrary truncation.

IMPORTANT SCOPE NOTE: there is no real bacterial/archaeal or eukaryotic
eDNA READ data in this project -- only the fish-marker ERR16922870 reads
exist, and Part 2 already established those reads are not 16S/18S at all
(Section 8). This script is therefore a HELD-OUT SYNTHETIC-FRAGMENT
validation of the architecture -- the same rigor Stage 0-4 used to
validate it for fish -- not a real-read classification run. Report it as
"architecture validated on Bacteria/Archaea and Eukaryota reference
data," never as "classified real bacterial/eukaryotic eDNA."

Held-out split / synthetic-fragment generation mirrors
stage0_diagnostics.build_holdout_split's methodology exactly (seed=42,
holdout_frac=0.2, N_TEST=1000, 118-140bp fragment lengths -- the real
ERR16922870 read-length distribution this project's methodology is built
around), generalized here to accept any reference table + embeddings
instead of MitoFish specifically.

Reuses BM25Index, mmr_select, rerank_candidates, and the exact production
hyperparameters from final_pipeline.py -- FROZEN, unmodified. Does not
touch final_pipeline.py, the MitoFish database, or any frozen Part 3/4
artifact.

Usage:
    python -m src.other_kingdom_diagnostics
"""
import csv
import os
import random
import time
import numpy as np

from .bm25_retrieval import BM25Index
from .mmr import mmr_select
from .nucleotide_rerank import rerank_candidates, align_pair
from .faiss_index import l2_normalize
from .final_pipeline import (
    BM25_K_MER, MMR_LAMBDA, MMR_POOL_SIZE, TOP_K,
    FROZEN_IDENTITY_THRESHOLD, BEST_GUESS_MIN_IDENTITY,
)

REAL_READ_LENGTHS = list(range(118, 141))
HOLDOUT_FRAC = 0.2
SEED = 42
N_TEST = 1000

KINGDOMS = [
    ("Bacteria_Archaea_16S_V4",
     "data/SILVA results/silva_16s_v4_sample.csv",
     "data/SILVA results/silva_16s_v4_ref_embeddings.npy",
     "data/SILVA results/silva_16s_v4_ref_ids.csv",
     "data/Other Kingdoms diagnostics/Bacteria_Archaea_16S_V4"),
    ("Eukaryota_18S_V9",
     "data/SILVA results/silva_18s_v9_sample.csv",
     "data/SILVA results/silva_18s_v9_ref_embeddings.npy",
     "data/SILVA results/silva_18s_v9_ref_ids.csv",
     "data/Other Kingdoms diagnostics/Eukaryota_18S_V9"),
]


def _load_reference_table(path):
    ids, seqs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])
    return ids, seqs


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Same >=95% full-sequence-identity ground truth as every prior
    stage, via nucleotide_rerank.align_pair (frozen, unmodified)."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def build_holdout_split(ref_table_csv, ref_emb_npy, ref_ids_csv, seed=SEED, holdout_frac=HOLDOUT_FRAC):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    ids, seqs = _load_reference_table(ref_table_csv)
    all_emb = np.load(ref_emb_npy).astype("float32")
    with open(ref_ids_csv) as f:
        emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
    assert emb_ids == ids, "ref_ids.csv order does not match reference table order"

    n = len(ids)
    perm = np_rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_ids = [ids[i] for i in train_idx]
    train_seqs = [seqs[i] for i in train_idx]
    train_emb = l2_normalize(all_emb[train_idx])

    holdout_idx = list(holdout_idx)[:N_TEST]
    synth_seqs, synth_true_seq = [], []
    for i in holdout_idx:
        seq = seqs[i]
        L = rng.choice(REAL_READ_LENGTHS)
        if len(seq) <= L:
            continue
        start = rng.randint(0, len(seq) - L)
        synth_seqs.append(seq[start:start + L])
        synth_true_seq.append(seq)

    return {
        "train_ids": train_ids, "train_seqs": train_seqs, "train_emb": train_emb,
        "synth_seqs": synth_seqs, "synth_true_seq": synth_true_seq,
    }


def run_kingdom(label, ref_table_csv, ref_emb_npy, ref_ids_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {label} ===")
    split = build_holdout_split(ref_table_csv, ref_emb_npy, ref_ids_csv)
    train_ids, train_seqs, train_emb = split["train_ids"], split["train_seqs"], split["train_emb"]
    synth_seqs, synth_true_seq = split["synth_seqs"], split["synth_true_seq"]
    n = len(synth_seqs)
    print(f"train={len(train_ids)}  holdout_fragments={n}")

    bm25 = BM25Index(k=BM25_K_MER)
    bm25.build(train_ids, train_seqs)
    train_id_to_seq = dict(zip(train_ids, train_seqs))

    earliest_hit_rank = []
    per_fragment_rows = []
    t0 = time.time()
    for i in range(n):
        pool = bm25.top_k(synth_seqs[i], k_results=MMR_POOL_SIZE)  # [(doc_idx, bm25_score), ...]
        if not pool:
            earliest_hit_rank.append(None)
            per_fragment_rows.append({"fragment_idx": i, "earliest_hit_rank": "", "identity_pct": 0.0,
                                       "decision": "unclassified", "correct_if_assigned": ""})
            continue

        doc_idxs = [d for d, _s in pool]
        scores = [s for _d, s in pool]
        emb_block = train_emb[doc_idxs]
        selected = mmr_select(doc_idxs, scores, emb_block, lam=MMR_LAMBDA, k_results=TOP_K)

        hit_rank = None
        for k, idx in enumerate(selected):
            if _is_hit(train_seqs[idx], synth_true_seq[i]):
                hit_rank = k + 1
                break
        earliest_hit_rank.append(hit_rank)

        score_by_idx = dict(pool)
        candidates = [(train_ids[idx], train_seqs[idx], score_by_idx[idx]) for idx in selected]
        reranked = rerank_candidates(synth_seqs[i], candidates)  # FROZEN, unmodified
        best = reranked[0]

        if best["identity_pct"] >= FROZEN_IDENTITY_THRESHOLD:
            decision = "assigned"
        elif best["identity_pct"] >= BEST_GUESS_MIN_IDENTITY:
            decision = "best_guess"
        else:
            decision = "best_guess_low_confidence"

        correct = ""
        if decision == "assigned":
            chosen_seq = train_id_to_seq.get(best["Reference_ID"])
            correct = int(chosen_seq is not None and _is_hit(chosen_seq, synth_true_seq[i]))

        per_fragment_rows.append({
            "fragment_idx": i, "earliest_hit_rank": hit_rank if hit_rank is not None else "",
            "identity_pct": best["identity_pct"], "decision": decision, "correct_if_assigned": correct,
        })

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{n}  ({time.time() - t0:.0f}s elapsed)")

    def recall_at(k):
        return sum(1 for r in earliest_hit_rank if r is not None and r <= k) / n

    n_assigned = sum(1 for r in per_fragment_rows if r["decision"] == "assigned")
    n_best_guess = sum(1 for r in per_fragment_rows if r["decision"] == "best_guess")
    n_low_conf = sum(1 for r in per_fragment_rows if r["decision"] == "best_guess_low_confidence")
    n_unclassified = sum(1 for r in per_fragment_rows if r["decision"] == "unclassified")
    n_correct_assigned = sum(1 for r in per_fragment_rows if r["decision"] == "assigned" and r["correct_if_assigned"] == 1)

    summary = {
        "kingdom": label,
        "n_holdout_fragments": n,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "n_assigned": n_assigned,
        "pct_assigned": n_assigned / n,
        "accuracy_on_assigned": (n_correct_assigned / n_assigned) if n_assigned else None,
        "n_best_guess": n_best_guess,
        "n_best_guess_low_confidence": n_low_conf,
        "n_unclassified_no_candidate": n_unclassified,
    }

    with open(f"{out_dir}/per_fragment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fragment_rows[0].keys()))
        w.writeheader()
        w.writerows(per_fragment_rows)

    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(f"=== {label}: BM25 -> MMR -> nucleotide reranker architecture, held-out validation ===\n")
        f.write("(SAME held-out methodology as Stage0-4/Task4 used for fish/MitoFish; SILVA-derived data from Explanation.md Part 2)\n\n")
        for k, v in summary.items():
            f.write(f"{k:28s}: {v}\n")

    print()
    for k, v in summary.items():
        print(f"{k:28s}: {v}")

    return summary


def main():
    print("=== Architecture validation on kingdoms other than fish (Bacteria/Archaea, Eukaryota) ===")
    print("Held-out synthetic-fragment methodology -- NOT a real-read classification (no real non-fish eDNA reads exist in this project).\n")

    all_summaries = []
    for label, ref_table_csv, ref_emb_npy, ref_ids_csv, out_dir in KINGDOMS:
        summary = run_kingdom(label, ref_table_csv, ref_emb_npy, ref_ids_csv, out_dir)
        all_summaries.append(summary)

    print("\n=== Summary ===")
    for s in all_summaries:
        print(f"{s['kingdom']:28s} recall@10={s['recall_at_10']:.3f} assigned={s['n_assigned']}/{s['n_holdout_fragments']} accuracy={s['accuracy_on_assigned']}")

    return all_summaries


if __name__ == "__main__":
    main()

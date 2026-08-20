"""
Calibrates an identity_pct threshold for the DNABERT top-K + nucleotide-
realignment pipeline (nucleotide_rerank.py), using the SAME held-out
sequence-identity methodology as mitofish_calibration.py (see that
module's docstring for why sequence-identity, not accession-identity, is
the correct ground truth for MitoFish -- many different accessions share
an identical sequence).

Held-out fragments are cut at the real read-length distribution
(118-140bp) from held-out full references, run through: (1) FAISS top-K
retrieval against a TRAIN-only index, (2) nucleotide-alignment rerank of
those K candidates. "Correct" means the chosen best-by-identity
candidate's FULL reference sequence is >=95% identical to the fragment's
true source sequence -- NOT exact string equality. A diagnostic run (see
Explanation.md Part 4) showed exact equality was a severe underestimate:
among "mismatched" high-identity picks, the chosen sequence was still
95-99.55% identical (median 99.0%) to the true source -- i.e. the same or
an extremely close relative in MitoFish's redundant accession set, not a
wrong answer. 95% is used as the correctness cutoff because it comfortably
separates that near-duplicate cluster (median 99%) from a real mismatch.
This directly measures the accuracy of the COMBINED method end-to-end,
not the FAISS step alone.

Usage:
    python -m src.mitofish_rerank_calibration
"""
import csv
import random
import sys
import numpy as np
import faiss

from . import config
from .embed_common import embed_batch
from .faiss_index import l2_normalize
from .nucleotide_rerank import rerank_candidates, align_pair
from .evaluate import sweep_thresholds, print_sweep

REAL_READ_LENGTHS = list(range(118, 141))
TOP_K = 10


def run(holdout_frac: float = 0.2, seed: int = 42, n_test: int = 1000):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    ids, seqs = [], []
    with open(config.REFERENCE_TABLE_CSV) as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])

    all_emb = np.load(config.REFERENCE_EMBEDDINGS_NPY).astype("float32")
    with open(config.REFERENCE_IDS_CSV) as f:
        emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
    assert emb_ids == ids

    n = len(ids)
    perm = np_rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_seqs = [seqs[i] for i in train_idx]
    train_ids = [ids[i] for i in train_idx]

    train_emb = l2_normalize(all_emb[train_idx])
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    holdout_idx = list(holdout_idx)[:n_test]
    synth_seqs, synth_true_seq = [], []
    for i in holdout_idx:
        seq = seqs[i]
        L = rng.choice(REAL_READ_LENGTHS)
        if len(seq) <= L:
            continue
        start = rng.randint(0, len(seq) - L)
        synth_seqs.append(seq[start:start + L])
        synth_true_seq.append(seq)

    synth_emb = []
    for i in range(0, len(synth_seqs), config.BATCH_SIZE):
        synth_emb.append(embed_batch(synth_seqs[i:i + config.BATCH_SIZE]))
        print(f"synthetic-read embedded {i + len(synth_emb[-1])}/{len(synth_seqs)}")
    synth_emb = l2_normalize(np.vstack(synth_emb).astype("float32"))

    sims, pos = index.search(synth_emb, TOP_K)

    id_to_seq = dict(zip(train_ids, train_seqs))  # last-wins on rare duplicate IDs; fine for this diagnostic

    best_identity_pct = []
    is_correct = []  # sequence-identity ground truth
    for i in range(len(synth_seqs)):
        candidates = [(train_ids[pos[i, k]], train_seqs[pos[i, k]], float(sims[i, k])) for k in range(TOP_K)]
        reranked = rerank_candidates(synth_seqs[i], candidates)
        best = reranked[0]
        best_identity_pct.append(best["identity_pct"])
        chosen_seq = id_to_seq.get(best["Reference_ID"])
        if chosen_seq == synth_true_seq[i]:
            correct = 1
        elif chosen_seq is not None:
            full_identity = align_pair(chosen_seq, synth_true_seq[i])["identity_pct"]
            correct = 1 if full_identity >= 95.0 else 0
        else:
            correct = 0
        is_correct.append(correct)
        if (i + 1) % 200 == 0:
            print(f"calibration reranked {i+1}/{len(synth_seqs)}")

    best_identity_pct = np.array(best_identity_pct)
    is_correct = np.array(is_correct)

    y_true = [1] * len(is_correct)
    y_pred = list(is_correct)  # 1 = correct, 0 = wrong; compare against constant "1" -> accuracy = mean(is_correct) on assigned subset
    thresholds = [round(t, 1) for t in np.arange(45, 100, 2.5)]
    rows = sweep_thresholds(y_true, y_pred, list(best_identity_pct), thresholds)
    print(f"=== MitoFish rerank calibration: identity_pct threshold sweep, n={len(is_correct)} ===")
    print(f"{'threshold':>10} {'%assigned':>10} {'accuracy':>10}")
    for r, t in zip(rows, thresholds):
        idx = [i for i, s in enumerate(best_identity_pct) if s >= t]
        acc = np.mean(is_correct[idx]) if idx else None
        pct = len(idx) / len(is_correct)
        acc_s = f"{acc:.3f}" if acc is not None else "NA"
        print(f"{t:>10} {pct*100:>9.1f}% {acc_s:>10}")
    return best_identity_pct, is_correct


if __name__ == "__main__":
    run()

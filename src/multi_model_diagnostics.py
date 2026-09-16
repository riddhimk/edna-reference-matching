"""
Model-vs-model comparison, round 2: HyenaDNA-tiny and GENA-LM
(bert-base-t2t) vs. the existing DNABERT-S/multimolecule pipeline, on the
IDENTICAL held-out split and synthetic fragments src/stage0_diagnostics.py
already established for DNABERT (same seed=42, holdout_frac=0.2, 118-140bp
fragment generation) -- reused via stage0_diagnostics.build_holdout_split,
UNMODIFIED, so Recall@K numbers here are directly comparable to both
data/Stage0 diagnostics/recall_diagnostic_summary.txt (DNABERT) and
data/NT diagnostics/nt_recall_summary.txt (Nucleotide Transformer).

Companion to src/nt_diagnostics.py -- same methodology, generalized to
loop over multiple models in one run instead of duplicating the script
per model. Ground truth ("is this candidate actually a hit") uses the
same >=95% full-sequence-identity check as every prior stage, via
nucleotide_rerank.align_pair (frozen, unmodified).

Does not modify, rerun, or recalibrate any existing stage, model, config,
or frozen Part 3/4 artifact. src/hyenadna_embed.py, src/genalm_embed.py,
and this script are the only new files touched by this comparison.

Usage:
    python -m src.multi_model_diagnostics
"""
import csv
import os
import time
import numpy as np
import faiss

from .stage0_diagnostics import build_holdout_split
from .nucleotide_rerank import align_pair
from .faiss_index import l2_normalize
from . import hyenadna_embed
from . import genalm_embed

MODELS = [
    ("HyenaDNA-tiny", hyenadna_embed, "data/HyenaDNA diagnostics"),
    ("GENA-LM", genalm_embed, "data/GENA-LM diagnostics"),
]
TOP_K = 10
BATCH_SIZE = 64


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Same >=95% full-sequence-identity ground truth as
    stage0_diagnostics.py / nt_diagnostics.py -- kept as an identical,
    self-contained copy for auditability."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def _embed_all(embed_module, seqs, label):
    vecs = []
    t0 = time.time()
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = embed_module.embed_batch(seqs[i:i + BATCH_SIZE])
        vecs.append(batch)
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  [{label}] embedded {i + len(batch)}/{len(seqs)}  ({time.time() - t0:.0f}s elapsed)")
    return np.vstack(vecs).astype("float32")


def run_one_model(model_label, embed_module, out_dir, split):
    train_ids, train_seqs = split["train_ids"], split["train_seqs"]
    synth_seqs, synth_true_seq = split["synth_seqs"], split["synth_true_seq"]

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {model_label} ===")

    print(f"Embedding {len(train_seqs)} reference (train) sequences...")
    train_emb = l2_normalize(_embed_all(embed_module, train_seqs, "refs"))

    print(f"Embedding {len(synth_seqs)} synthetic holdout fragments...")
    synth_emb = l2_normalize(_embed_all(embed_module, synth_seqs, "frags"))

    index = faiss.IndexFlatIP(embed_module.EMBEDDING_DIM)
    index.add(train_emb)
    _sims, pos = index.search(synth_emb, TOP_K)

    earliest_hit_rank = []
    for i in range(len(synth_seqs)):
        hit_rank = None
        for k in range(TOP_K):
            if _is_hit(train_seqs[pos[i, k]], synth_true_seq[i]):
                hit_rank = k + 1
                break
        earliest_hit_rank.append(hit_rank)
        if (i + 1) % 200 == 0:
            print(f"  recall-checked {i + 1}/{len(synth_seqs)}")

    n = len(earliest_hit_rank)

    def recall_at(k):
        return sum(1 for r in earliest_hit_rank if r is not None and r <= k) / n

    n_absent = sum(1 for r in earliest_hit_rank if r is None)

    summary = {
        "model": model_label,
        "n_holdout_fragments": n,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "n_absent_from_top10": n_absent,
        "pct_absent_from_top10": n_absent / n,
    }

    np.save(f"{out_dir}/train_embeddings.npy", train_emb)
    with open(f"{out_dir}/train_ids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reference_ID"])
        for rid in train_ids:
            w.writerow([rid])

    with open(f"{out_dir}/recall_summary.txt", "w") as f:
        f.write(f"=== {model_label} Recall@K ===\n")
        f.write("(IDENTICAL held-out split/fragments as data/Stage0 diagnostics/ and data/NT diagnostics/)\n\n")
        for k, v in summary.items():
            f.write(f"{k:28s}: {v}\n")

    print()
    for k, v in summary.items():
        print(f"{k:28s}: {v}")

    return summary


def main():
    print("=== Model comparison round 2: HyenaDNA-tiny, GENA-LM vs DNABERT-S / Nucleotide Transformer ===")
    print("Held-out split/fragments reused unmodified from stage0_diagnostics.build_holdout_split.")

    split = build_holdout_split()  # text-only reuse; this split's own DNABERT embeddings are ignored here

    all_summaries = []
    for model_label, embed_module, out_dir in MODELS:
        summary = run_one_model(model_label, embed_module, out_dir, split)
        all_summaries.append(summary)

    print("\n=== Summary ===")
    for s in all_summaries:
        print(f"{s['model']:16s} recall@1={s['recall_at_1']:.3f} @5={s['recall_at_5']:.3f} @10={s['recall_at_10']:.3f}")

    return all_summaries


if __name__ == "__main__":
    main()

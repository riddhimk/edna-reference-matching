"""
Does the fine-tuned embedding model actually change END-TO-END pipeline
results? Parts 11-12 measured the fine-tuned bacteria model with
embedding-only retrieval (plain FAISS cosine, Recall@10 17.1% -> 79.2%),
but that is NOT how the model is used in production: BM25 performs the
primary retrieval and the embeddings only supply MMR's diversity signal.
A better embedding model could therefore change very little end-to-end --
except under RRF, where the embedding model contributes a full ranked
list of its own and a stronger model should matter a great deal.

This script answers that by running the same A/B/C/D architecture
comparison as src/final_evaluation.py (Task 4), but on the bacteria
held-out split and with BOTH embedding sets, so the only thing that
differs between paired arms is frozen vs. fine-tuned embeddings:

    A.  BM25 (k=6)                      -> reranker   [embedding-independent]
    B.  BM25 -> MMR (lambda=0.5)        -> reranker   [frozen | fine-tuned]
    C.  DNABERT alone                   -> reranker   [frozen | fine-tuned]
    D.  RRF (DNABERT + BM25, k_const=10)-> reranker   [frozen | fine-tuned]

Architecture A appears once because BM25 is lexical and uses no
embeddings at all -- it is the invariant reference point against which
both embedding sets are judged.

Everything frozen stays frozen: BM25Index, mmr_select, fuse_top_k and
nucleotide_rerank.rerank_candidates are imported and called unmodified,
the decision threshold is final_pipeline's calibrated 95% identity, and
the held-out split comes from the same seed-42 function the fine-tuning
itself excluded from training (so the fine-tuned model has never seen
these fragments' source references).

Usage:
    python -m src.finetuned_architecture_evaluation [kingdom] [finetuned_model_dir]
    kingdom in {bacteria, fish, eukaryota}, defaults to bacteria for
    backward compatibility with the original (bacteria-only) invocation.
"""
import csv
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from .bm25_retrieval import BM25Index
from .mmr import mmr_select
from .rrf_fusion import fuse_top_k
from .nucleotide_rerank import rerank_candidates, align_pair
from .faiss_index import l2_normalize
from .final_pipeline import (
    BM25_K_MER, MMR_LAMBDA, MMR_POOL_SIZE, TOP_K, FROZEN_IDENTITY_THRESHOLD,
)
from .mmr_lambda_sweep import redundancy_of_sets

torch.set_num_threads(os.cpu_count() or 8)

DEFAULT_FINETUNED_MODEL_DIR = {
    "bacteria": "models/dnaberts_finetuned_bacteria_v2",
    "fish": "models/dnaberts_finetuned_fish_v2",
    "eukaryota": "models/dnaberts_finetuned_eukaryota_v2",
}
DEFAULT_OUT_DIR = {
    "bacteria": "data/Finetuned architecture evaluation",
    "fish": "data/Finetuned architecture evaluation/fish",
    "eukaryota": "data/Finetuned architecture evaluation/eukaryota",
}
RRF_K_CONST = 10          # Stage 2's selected value -- not re-swept here
EMBED_BATCH = 32


def get_split(kingdom: str):
    """Mirrors evaluate_finetuned.get_split() so this script measures the
    IDENTICAL held-out split each kingdom's fine-tuned checkpoint was
    trained/evaluated against."""
    if kingdom == "fish":
        from .stage0_diagnostics import build_holdout_split
        return build_holdout_split()
    from .other_kingdom_diagnostics import build_holdout_split
    from .evaluate_finetuned import REF_TABLES
    ref_table_csv, ref_emb_npy, ref_ids_csv = REF_TABLES[kingdom]
    return build_holdout_split(ref_table_csv, ref_emb_npy, ref_ids_csv)


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def _embed_all(embed_fn, seqs, label):
    vecs = []
    t0 = time.time()
    for i in range(0, len(seqs), EMBED_BATCH):
        vecs.append(embed_fn(seqs[i:i + EMBED_BATCH]))
        if (i // EMBED_BATCH) % 40 == 0:
            print(f"  [{label}] {i + len(vecs[-1])}/{len(seqs)}  ({time.time() - t0:.0f}s)")
    return np.vstack(vecs).astype("float32")


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


def embedding_rank_lists(query_emb, ref_emb, top_k=TOP_K) -> List[List[int]]:
    """Cosine-similarity rank order (both inputs L2-normalized)."""
    import faiss
    index = faiss.IndexFlatIP(ref_emb.shape[1])
    index.add(ref_emb)
    _sims, pos = index.search(query_emb, top_k)
    return [[int(x) for x in pos[i]] for i in range(pos.shape[0])]


def evaluate_arm(name: str, per_fragment_lists: List[List[int]], per_fragment_scores: List[Dict[int, float]],
                 train_ids, train_seqs, synth_seqs, synth_true_seq, redundancy_emb, train_id_to_seq):
    n = len(synth_seqs)
    t0 = time.time()

    earliest_ranks = []
    best_identity, is_correct = [], []
    for i in range(n):
        doc_idxs = per_fragment_lists[i]

        hit_rank = None
        for k, idx in enumerate(doc_idxs[:TOP_K]):
            if _is_hit(train_seqs[idx], synth_true_seq[i]):
                hit_rank = k + 1
                break
        earliest_ranks.append(hit_rank)

        if not doc_idxs:
            best_identity.append(0.0)
            is_correct.append(0)
            continue

        scores = per_fragment_scores[i]
        candidates = [(train_ids[idx], train_seqs[idx], scores.get(idx, 0.0)) for idx in doc_idxs]
        best = rerank_candidates(synth_seqs[i], candidates)[0]  # FROZEN, unmodified
        best_identity.append(best["identity_pct"])
        chosen_seq = train_id_to_seq.get(best["Reference_ID"])
        is_correct.append(1 if (chosen_seq is not None and _is_hit(chosen_seq, synth_true_seq[i])) else 0)

        if (i + 1) % 250 == 0:
            print(f"  [{name}] reranked {i + 1}/{n}  ({time.time() - t0:.0f}s)")

    best_identity = np.array(best_identity)
    is_correct = np.array(is_correct)
    assigned = best_identity >= FROZEN_IDENTITY_THRESHOLD
    n_assigned = int(assigned.sum())

    def recall_at(k):
        return sum(1 for r in earliest_ranks if r is not None and r <= k) / n

    dist, _set_level = redundancy_of_sets(per_fragment_lists, redundancy_emb)

    result = {
        "architecture": name,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "n_assigned": n_assigned,
        "pct_assigned": n_assigned / n,
        "accuracy_on_assigned": float(is_correct[assigned].mean()) if n_assigned else None,
        "mean_pairwise_sim": dist["mean"],
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  [{name}] recall@1={result['recall_at_1']:.3f} @10={result['recall_at_10']:.3f} "
          f"assigned={n_assigned}/{n} acc={result['accuracy_on_assigned']}")
    return result


def main(kingdom: str = "bacteria", finetuned_model_dir: str = None, out_dir: str = None):
    finetuned_model_dir = finetuned_model_dir or DEFAULT_FINETUNED_MODEL_DIR[kingdom]
    out_dir = out_dir or DEFAULT_OUT_DIR[kingdom]
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Does fine-tuning change END-TO-END pipeline results? ({kingdom}) ===")
    print("A: BM25 | B: BM25->MMR | C: DNABERT alone | D: RRF(DNABERT+BM25) -- each through the frozen reranker\n")

    split = get_split(kingdom)

    train_ids, train_seqs = split["train_ids"], split["train_seqs"]
    synth_seqs, synth_true_seq = split["synth_seqs"], split["synth_true_seq"]
    frozen_ref_emb = split["train_emb"]  # already L2-normalized frozen DNABERT embeddings
    train_id_to_seq = dict(zip(train_ids, train_seqs))
    n = len(synth_seqs)
    print(f"train={len(train_ids)}  holdout_fragments={n}\n")

    # --- embeddings: frozen (refs already on disk) and fine-tuned (both sides) ---
    from .embed_common import embed_batch as frozen_embed
    print("Embedding holdout fragments with FROZEN model...")
    frozen_query_emb = l2_normalize(_embed_all(frozen_embed, synth_seqs, "frozen-frags"))

    ft_embed = load_finetuned_embed_fn(finetuned_model_dir)
    print("Embedding references with FINE-TUNED model...")
    ft_ref_emb = l2_normalize(_embed_all(ft_embed, train_seqs, "ft-refs"))
    print("Embedding holdout fragments with FINE-TUNED model...")
    ft_query_emb = l2_normalize(_embed_all(ft_embed, synth_seqs, "ft-frags"))

    # --- BM25 (shared by A, B, D; embedding-independent) ---
    print("\nBuilding BM25 index...")
    bm25 = BM25Index(k=BM25_K_MER)
    bm25.build(train_ids, train_seqs)
    bm25_pool = [bm25.top_k(s, k_results=MMR_POOL_SIZE) for s in synth_seqs]
    bm25_lists = [[d for d, _s in pool[:TOP_K]] for pool in bm25_pool]
    bm25_scores = [{d: s for d, s in pool} for pool in bm25_pool]

    # --- embedding-derived rank lists ---
    print("Computing embedding rank lists...")
    frozen_dnabert_lists = embedding_rank_lists(frozen_query_emb, frozen_ref_emb)
    ft_dnabert_lists = embedding_rank_lists(ft_query_emb, ft_ref_emb)
    zero_scores = [{} for _ in range(n)]  # reranker ignores the 3rd tuple slot

    def mmr_lists(ref_emb):
        out = []
        for i in range(n):
            doc_idxs = [d for d, _s in bm25_pool[i]]
            scores = [s for _d, s in bm25_pool[i]]
            if not doc_idxs:
                out.append([])
                continue
            out.append(mmr_select(doc_idxs, scores, ref_emb[doc_idxs], lam=MMR_LAMBDA, k_results=TOP_K))
        return out

    def rrf_lists(dnabert_lists):
        return [fuse_top_k([dnabert_lists[i], bm25_lists[i]], k_const=RRF_K_CONST, k_results=TOP_K)
                for i in range(n)]

    arms = [
        ("A_BM25", bm25_lists, bm25_scores, frozen_ref_emb),
        ("B_BM25_MMR_frozen", mmr_lists(frozen_ref_emb), bm25_scores, frozen_ref_emb),
        ("B_BM25_MMR_finetuned", mmr_lists(ft_ref_emb), bm25_scores, ft_ref_emb),
        ("C_DNABERT_frozen", frozen_dnabert_lists, zero_scores, frozen_ref_emb),
        ("C_DNABERT_finetuned", ft_dnabert_lists, zero_scores, ft_ref_emb),
        ("D_RRF_frozen", rrf_lists(frozen_dnabert_lists), bm25_scores, frozen_ref_emb),
        ("D_RRF_finetuned", rrf_lists(ft_dnabert_lists), bm25_scores, ft_ref_emb),
    ]

    results = []
    for name, lists_, scores_, red_emb in arms:
        print(f"\n--- {name} ---")
        results.append(evaluate_arm(name, lists_, scores_, train_ids, train_seqs,
                                    synth_seqs, synth_true_seq, red_emb, train_id_to_seq))

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump({"kingdom": kingdom, "n_holdout_fragments": n,
                   "finetuned_model": finetuned_model_dir,
                   "identity_threshold": FROZEN_IDENTITY_THRESHOLD,
                   "rrf_k_const": RRF_K_CONST, "architectures": results}, f, indent=2)
    with open(f"{out_dir}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\n=== Summary ({kingdom}, n={n} held-out fragments) ===")
    print(f"{'architecture':26s} {'R@1':>7s} {'R@10':>7s} {'assigned':>9s} {'accuracy':>9s}")
    for r in results:
        acc = f"{r['accuracy_on_assigned']:.3f}" if r["accuracy_on_assigned"] is not None else "NA"
        print(f"{r['architecture']:26s} {r['recall_at_1']:7.3f} {r['recall_at_10']:7.3f} "
              f"{r['pct_assigned']:9.3f} {acc:>9s}")
    print("\nOutputs written to:", out_dir)
    return results


if __name__ == "__main__":
    _kingdom = sys.argv[1] if len(sys.argv) > 1 else "bacteria"
    _model_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(_kingdom, _model_dir)

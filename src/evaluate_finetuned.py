"""
Evaluates each kingdom's fine-tuned DNABERT-S checkpoint (src/finetune_dnabert.py)
against the FROZEN baseline Recall@K numbers already established for the
unmodified pretrained model, on the IDENTICAL held-out split/synthetic
fragments used to measure those baselines:

  - Fish: stage0_diagnostics.build_holdout_split() -- same split that
    produced data/Stage0 diagnostics/recall_diagnostic_summary.txt
    (DNABERT Recall@10 = 58.2%).
  - Bacteria/Archaea, Eukaryota: other_kingdom_diagnostics.build_holdout_split()
    -- same split that produced data/Other Kingdoms diagnostics/*/summary.txt
    (Recall@10 = 82.0% / 70.5%).

Only the embedding model differs (fine-tuned checkpoint vs. frozen
multimolecule/dnaberts); the split, fragment generation, FAISS index
construction, and >=95%-identity ground truth (nucleotide_rerank.align_pair)
are reused UNCHANGED, so any Recall@K difference is attributable to the
fine-tuning, not a different test set.

Does not modify final_pipeline.py, the frozen baseline diagnostics, or
any existing checkpoint/config.

Usage:
    python -m src.evaluate_finetuned
"""
import time
import numpy as np
import faiss

from .faiss_index import l2_normalize
from .nucleotide_rerank import align_pair

# Fish: legitimately reusable as-is -- src/finetune_dnabert.py fine-tuned on a
# 5,000-record SUBSAMPLE of this exact same full MitoFish table (same region,
# same primers, same length distribution), so this split is in-domain for both
# the frozen and fine-tuned fish model. See data/Stage0 diagnostics/.
#
# Bacteria/Eukaryota: NOT reused from data/Other Kingdoms diagnostics/ (those
# numbers were measured on Part 2's DIFFERENT primer extraction --
# 16s_v4/18s_v9 rather than the 16s_v4v5_tara/18s_v9_tara tables the
# fine-tuned models actually trained on). Recomputed fresh below, on the
# IDENTICAL split used for the fine-tuned evaluation, for a fair comparison.
FROZEN_BASELINE_RECALL_AT_10 = {
    "fish": 0.582,
}

KINGDOMS = [
    ("fish", "models/dnaberts_finetuned_fish"),
    ("bacteria", "models/dnaberts_finetuned_bacteria"),
    ("eukaryota", "models/dnaberts_finetuned_eukaryota"),
]

MAX_LENGTH = 300
BATCH_SIZE = 32


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def load_finetuned_embed_fn(model_dir: str):
    import multimolecule  # noqa: F401
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()

    def embed_batch(seqs):
        enc = tok(seqs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
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
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = embed_fn(seqs[i:i + BATCH_SIZE])
        vecs.append(batch)
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  [{label}] embedded {i + len(batch)}/{len(seqs)}  ({time.time() - t0:.0f}s elapsed)")
    return np.vstack(vecs).astype("float32")


REF_TABLES = {
    # MUST match exactly what src/finetune_dnabert.py fine-tuned each model on --
    # evaluating against a different primer/region's split would silently
    # compare in-domain (fish) against out-of-domain (bacteria/eukaryota)
    # performance, which is what the first version of this script did by
    # mistake (see Explanation.md Part 11, Section on this bug).
    "bacteria": (
        "data/SILVA results/silva_16s_v4v5_tara_reference_table.csv",
        "data/SILVA results/silva_16s_v4v5_tara_ref_embeddings.npy",
        "data/SILVA results/silva_16s_v4v5_tara_ref_ids.csv",
    ),
    "eukaryota": (
        "data/SILVA results/silva_18s_v9_tara_reference_table.csv",
        "data/SILVA results/silva_18s_v9_tara_ref_embeddings.npy",
        "data/SILVA results/silva_18s_v9_tara_ref_ids.csv",
    ),
}


def get_split(kingdom: str):
    if kingdom == "fish":
        from .stage0_diagnostics import build_holdout_split
        split = build_holdout_split()
        return split["train_ids"], split["train_seqs"], split["synth_seqs"], split["synth_true_seq"]
    else:
        from .other_kingdom_diagnostics import build_holdout_split
        ref_table_csv, ref_emb_npy, ref_ids_csv = REF_TABLES[kingdom]
        split = build_holdout_split(ref_table_csv, ref_emb_npy, ref_ids_csv)
        return split["train_ids"], split["train_seqs"], split["synth_seqs"], split["synth_true_seq"]


def _recall_from_embeddings(train_emb, train_seqs, synth_emb, synth_seqs, synth_true_seq, top_k=10):
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)
    _sims, pos = index.search(synth_emb, top_k)

    n = len(synth_seqs)
    earliest_hit_rank = []
    for i in range(n):
        hit_rank = None
        for k in range(top_k):
            if _is_hit(train_seqs[pos[i, k]], synth_true_seq[i]):
                hit_rank = k + 1
                break
        earliest_hit_rank.append(hit_rank)

    def recall_at(k):
        return sum(1 for r in earliest_hit_rank if r is not None and r <= k) / n

    return {"recall_at_1": recall_at(1), "recall_at_5": recall_at(5), "recall_at_10": recall_at(10)}


def evaluate_kingdom(kingdom: str, model_dir: str):
    print(f"\n=== {kingdom} ===")
    train_ids, train_seqs, synth_seqs, synth_true_seq = get_split(kingdom)
    print(f"train={len(train_ids)}  holdout_fragments={len(synth_seqs)}")

    finetuned_embed_fn = load_finetuned_embed_fn(model_dir)
    print("Embedding reference (train) sequences with FINE-TUNED model...")
    train_emb_ft = l2_normalize(_embed_all(finetuned_embed_fn, train_seqs, "refs-ft"))
    print("Embedding synthetic holdout fragments with FINE-TUNED model...")
    synth_emb_ft = l2_normalize(_embed_all(finetuned_embed_fn, synth_seqs, "frags-ft"))
    ft_recall = _recall_from_embeddings(train_emb_ft, train_seqs, synth_emb_ft, synth_seqs, synth_true_seq)

    if kingdom in FROZEN_BASELINE_RECALL_AT_10:
        frozen_recall_at_10 = FROZEN_BASELINE_RECALL_AT_10[kingdom]
    else:
        from .embed_common import embed_batch as frozen_embed_fn
        print("Embedding reference (train) sequences with FROZEN baseline model (same split, fair comparison)...")
        train_emb_fr = l2_normalize(_embed_all(frozen_embed_fn, train_seqs, "refs-frozen"))
        print("Embedding synthetic holdout fragments with FROZEN baseline model...")
        synth_emb_fr = l2_normalize(_embed_all(frozen_embed_fn, synth_seqs, "frags-frozen"))
        frozen_recall = _recall_from_embeddings(train_emb_fr, train_seqs, synth_emb_fr, synth_seqs, synth_true_seq)
        frozen_recall_at_10 = frozen_recall["recall_at_10"]

    result = {
        "kingdom": kingdom,
        "n_holdout_fragments": len(synth_seqs),
        "recall_at_1_finetuned": ft_recall["recall_at_1"],
        "recall_at_5_finetuned": ft_recall["recall_at_5"],
        "recall_at_10_finetuned": ft_recall["recall_at_10"],
        "recall_at_10_frozen_baseline": frozen_recall_at_10,
        "delta_recall_at_10": ft_recall["recall_at_10"] - frozen_recall_at_10,
    }
    for k, v in result.items():
        print(f"  {k:32s}: {v}")
    return result


def main():
    print("=== Fine-tuned vs frozen-baseline Recall@K, per kingdom ===")
    print("(identical held-out split/fragments as each kingdom's already-established frozen baseline)\n")
    results = [evaluate_kingdom(k, d) for k, d in KINGDOMS]

    print("\n=== Summary ===")
    for r in results:
        sign = "+" if r["delta_recall_at_10"] >= 0 else ""
        print(f"{r['kingdom']:10s} frozen={r['recall_at_10_frozen_baseline']:.3f} "
              f"finetuned={r['recall_at_10_finetuned']:.3f} delta={sign}{r['delta_recall_at_10']:.3f}")
    return results


if __name__ == "__main__":
    main()

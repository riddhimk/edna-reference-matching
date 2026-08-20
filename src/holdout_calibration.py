"""
Stage 12 driver -- threshold calibration using a SILVA-internal holdout.

Honesty note: the real experimental reads (ERR16922870) have no independently
verified species-level ground truth available in this project, so
evaluate.py's sweep cannot be run against them directly (see PDF Section 12 /
17: "this is the part that still needs to be implemented and experimentally
validated"). Instead, this holds out a fraction of the SILVA reference
sample itself, whose taxonomy IS known (it's the label SILVA ships), and
asks: "if a held-out reference sequence is queried against the rest of the
index, does the nearest hit's taxonomy match the held-out sequence's own
taxonomy, and at what similarity does that stop being reliable?"

This is a real, computed threshold sweep -- not a guessed constant -- but it
is a proxy calibration (full-length reference vs. full-length reference),
not a validation against the actual short experimental reads, which remain
unlabeled. Report it as such.

Usage:
    python -m src.holdout_calibration
"""
import csv
import sys
import numpy as np
import faiss

from . import config
from .faiss_index import l2_normalize
from .evaluate import sweep_thresholds, print_sweep


def taxonomy_at_depth(tax_full: str, depth: int) -> str:
    fields = [f.strip() for f in tax_full.split(";") if f.strip()]
    return ";".join(fields[:depth])


def run(depth: int = 3, holdout_frac: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)

    ref_emb = np.load(config.REFERENCE_EMBEDDINGS_NPY).astype("float32")
    ref_emb = l2_normalize(ref_emb)

    ids, tax_full = [], []
    with open(config.REFERENCE_IDS_CSV) as f:
        id_rows = list(csv.DictReader(f))
    ref_id_to_row = {}
    with open(config.REFERENCE_TABLE_CSV) as f:
        for row in csv.DictReader(f):
            ref_id_to_row[row["Reference_ID"]] = row["Taxonomy_Full"]
    for row in id_rows:
        ids.append(row["Reference_ID"])
        tax_full.append(ref_id_to_row.get(row["Reference_ID"], ""))

    n = ref_emb.shape[0]
    assert n == len(ids), f"embedding count {n} != id count {len(ids)}"
    perm = rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    train_emb = ref_emb[train_idx]
    holdout_emb = ref_emb[holdout_idx]

    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    sims, pos = index.search(holdout_emb, 1)
    sims = sims[:, 0]
    pos = pos[:, 0]

    y_true = [taxonomy_at_depth(tax_full[i], depth) for i in holdout_idx]
    y_pred = [taxonomy_at_depth(tax_full[train_idx[p]], depth) for p in pos]

    thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.02)]
    rows = sweep_thresholds(y_true, y_pred, list(sims), thresholds)
    print(f"=== Holdout threshold sweep (n_train={len(train_idx)}, n_holdout={len(holdout_idx)}, taxonomy_depth={depth}) ===")
    print_sweep(rows)
    return rows, y_true, y_pred, sims


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else config.CLUSTER_LABEL_TAXONOMY_DEPTH
    run(depth=depth)

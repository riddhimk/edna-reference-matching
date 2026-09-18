"""
Fine-tunes DNABERT-S (multimolecule/dnaberts) SEPARATELY on each
kingdom's real reference data (1,000-5,000 records per kingdom, not the
full database), using a supervised contrastive objective so sequences
from the same species/genus/taxon embed closer together in the output
space -- the same species-aware training philosophy DNABERT-S's own
paper uses (contrastive, not masked-language-modeling), specialized per
kingdom instead of left as the frozen general-purpose checkpoint.

Why supervised contrastive, not continued MLM pretraining: every
downstream use of these embeddings (FAISS cosine retrieval, MMR
diversity scoring) is nearest-neighbor geometry, so directly optimizing
"same class close, different class far" is a more direct match to that
use case than optimizing token-prediction accuracy.

Labels, per kingdom (see the calling scripts for exactly which real
subsample each kingdom uses):
  - Bacteria/Archaea, Eukaryota: SILVA's own Taxonomy_Full field,
    truncated to a fixed depth -- these reference tables already carry
    real taxonomy from the original SILVA FASTA headers (label_kind=
    "taxonomy_depth").
  - Fish (MitoFish): MitoFish ships NO species names at all (established
    in Explanation.md Part 3). Real species names for this specific
    subsample must be fetched from NCBI first (src/species_lookup.py)
    and written into the CSV's Organism_Label column BEFORE calling this
    script; here we group by genus, the first word of that resolved name
    (label_kind="genus").

Sequences whose class has fewer than 2 members are dropped from training
(no positive pair possible for a class-balanced batch) -- reported, not
silently discarded.

Does not modify src/embed_common.py, src/config.py, or any frozen
artifact; saves a new fine-tuned model checkpoint to its own output
directory per kingdom instead.

Usage:
    python -m src.finetune_dnabert <ref_table_csv> <label_kind> <out_model_dir> [taxonomy_depth]
    label_kind in {taxonomy_depth, genus}
"""
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import List

import numpy as np
import torch

MODEL_NAME = "multimolecule/dnaberts"
MAX_LENGTH = 300
BATCH_CLASSES = 4          # distinct classes per batch
BATCH_PER_CLASS = 4        # sequences per class per batch (so batch size = 16)
EPOCHS = 3
LEARNING_RATE = 1e-5
TEMPERATURE = 0.1
SEED = 42

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    import multimolecule  # noqa: F401 -- registers multimolecule/* architectures
    from transformers import AutoTokenizer, AutoModel

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)


def load_rows(ref_table_csv: str):
    with open(ref_table_csv) as f:
        return list(csv.DictReader(f))


def build_labels(rows: List[dict], label_kind: str, taxonomy_depth: int = 4) -> List[str]:
    if label_kind == "taxonomy_depth":
        return [";".join(r["Taxonomy_Full"].split(";")[:taxonomy_depth]) for r in rows]
    elif label_kind == "genus":
        return [r["Organism_Label"].split()[0] if r.get("Organism_Label") else "" for r in rows]
    else:
        raise ValueError(f"Unknown label_kind {label_kind!r}")


def filter_multi_member_classes(seqs: List[str], labels: List[str]):
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    kept_seqs, kept_labels = [], []
    for seq, lab in zip(seqs, labels):
        if lab and counts[lab] >= 2:
            kept_seqs.append(seq)
            kept_labels.append(lab)
    return kept_seqs, kept_labels, counts


class ClassBalancedBatcher:
    """Each batch: BATCH_CLASSES distinct classes, BATCH_PER_CLASS sequences
    each (sampled with replacement if a class is smaller than that), so
    every anchor has at least one true positive in its own batch."""

    def __init__(self, seqs, labels, rng: random.Random):
        self.seqs = seqs
        self.rng = rng
        self.by_class = defaultdict(list)
        for i, lab in enumerate(labels):
            self.by_class[lab].append(i)
        self.classes = list(self.by_class.keys())

    def steps_per_epoch(self) -> int:
        return max(1, len(self.seqs) // (BATCH_CLASSES * BATCH_PER_CLASS))

    def sample_batch(self):
        chosen_classes = self.rng.sample(self.classes, min(BATCH_CLASSES, len(self.classes)))
        batch_idx, batch_labels = [], []
        for c in chosen_classes:
            members = self.by_class[c]
            picks = self.rng.choices(members, k=BATCH_PER_CLASS) if len(members) < BATCH_PER_CLASS \
                else self.rng.sample(members, BATCH_PER_CLASS)
            batch_idx.extend(picks)
            batch_labels.extend([c] * BATCH_PER_CLASS)
        return [self.seqs[i] for i in batch_idx], batch_labels


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: List[str], temperature: float) -> torch.Tensor:
    """SupCon (Khosla et al. 2020), batch formulation. embeddings: (N, D),
    assumed L2-normalized. labels: length-N list of class labels."""
    n = embeddings.shape[0]
    sim = embeddings @ embeddings.T / temperature  # (N, N)
    labels_arr = np.array(labels)
    same_class = torch.tensor((labels_arr[:, None] == labels_arr[None, :]), dtype=torch.bool)
    self_mask = torch.eye(n, dtype=torch.bool)
    positive_mask = same_class & ~self_mask
    valid_mask = ~self_mask

    sim_masked = sim.masked_fill(~valid_mask, float("-inf"))
    log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)

    n_pos = positive_mask.sum(dim=1).clamp(min=1)
    loss_per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)) / n_pos
    has_positive = positive_mask.sum(dim=1) > 0
    return loss_per_anchor[has_positive].mean()


def finetune(ref_table_csv: str, label_kind: str, out_model_dir: str, taxonomy_depth: int = 4,
             epochs: int = EPOCHS, seed: int = SEED):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    rows = load_rows(ref_table_csv)
    seqs = [r["Sequence"] for r in rows]
    labels = build_labels(rows, label_kind, taxonomy_depth)
    kept_seqs, kept_labels, class_counts = filter_multi_member_classes(seqs, labels)

    n_classes_total = len({lab for lab in labels if lab})
    n_classes_kept = len({lab for lab in kept_labels})
    print(f"Loaded {len(rows)} records, {n_classes_total} distinct classes ({label_kind}, depth={taxonomy_depth})")
    print(f"Usable after dropping singleton classes: {len(kept_seqs)}/{len(rows)} records, {n_classes_kept} classes")

    _load_model()
    _model.train()
    opt = torch.optim.AdamW(_model.parameters(), lr=LEARNING_RATE)

    batcher = ClassBalancedBatcher(kept_seqs, kept_labels, rng)
    steps_per_epoch = batcher.steps_per_epoch()
    print(f"Training: {epochs} epochs x {steps_per_epoch} steps/epoch, batch={BATCH_CLASSES}x{BATCH_PER_CLASS}")

    loss_log = []
    t0 = time.time()
    for epoch in range(epochs):
        epoch_losses = []
        for step in range(steps_per_epoch):
            batch_seqs, batch_labels = batcher.sample_batch()
            enc = _tokenizer(batch_seqs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            out = _model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=1)

            loss = supervised_contrastive_loss(pooled, batch_labels, TEMPERATURE)
            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_losses.append(float(loss.item()))
            if (step + 1) % 20 == 0:
                print(f"  epoch {epoch + 1}/{epochs} step {step + 1}/{steps_per_epoch} "
                      f"loss={float(loss.item()):.4f}  ({time.time() - t0:.0f}s elapsed)")

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_log.append(mean_epoch_loss)
        print(f"epoch {epoch + 1}/{epochs} mean loss: {mean_epoch_loss:.4f}")

    os.makedirs(out_model_dir, exist_ok=True)
    _model.eval()
    _model.save_pretrained(out_model_dir)
    _tokenizer.save_pretrained(out_model_dir)

    metadata = {
        "base_model": MODEL_NAME,
        "ref_table_csv": ref_table_csv,
        "label_kind": label_kind,
        "taxonomy_depth": taxonomy_depth if label_kind == "taxonomy_depth" else None,
        "n_records_total": len(rows),
        "n_records_used": len(kept_seqs),
        "n_classes_used": n_classes_kept,
        "epochs": epochs,
        "batch_classes": BATCH_CLASSES,
        "batch_per_class": BATCH_PER_CLASS,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "mean_loss_per_epoch": loss_log,
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_model_dir, "finetune_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved fine-tuned model to {out_model_dir}")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    ref_table_csv, label_kind, out_model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    n_epochs = int(sys.argv[5]) if len(sys.argv) > 5 else EPOCHS
    finetune(ref_table_csv, label_kind, out_model_dir, taxonomy_depth=depth, epochs=n_epochs)

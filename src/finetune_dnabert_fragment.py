"""
Corrected fine-tuning for the bacteria regression found in Part 11
(Recall@10 17.1% frozen -> 7.9% fine-tuned). Three concrete, separately
confirmed causes were diagnosed before changing anything -- see
Explanation.md Part 12 -- and this module fixes all three. Kept as a
separate module from src/finetune_dnabert.py so Part 11's already-run,
already-documented v1 results stay reproducible.

Cause 1 -- INPUT MISMATCH (the big one, and specific to bacteria).
At evaluation (and in real use) the query is a 118-140bp read, while the
reference database holds full amplicons. For bacteria those amplicons
average 373bp, so a query covers only ~35% of a reference -- versus 64%
for fish and 99% for eukaryota. v1 trained exclusively on FULL amplicons,
so the model never saw a short fragment during training and had no reason
to place a fragment near its own parent. The two kingdoms whose
references are nearly query-length (eukaryota) or close to it (fish) were
barely affected; bacteria, with the worst asymmetry, broke.
  Fix: every training pair is (random 118-140bp fragment) <-> (its own
  full amplicon), so the model directly optimizes the fragment->full
  retrieval it is actually asked to perform.

Cause 2 -- LABEL GRANULARITY. v1 used SILVA Taxonomy_Full truncated to
depth 4, which for bacteria is order level: the largest single class held
418 sequences, and the top five classes held 343-418 each. The evaluation
is an INSTANCE retrieval task (find a reference >=95% identical to the
fragment's true source, inside the top 10), so training the model to make
418 different sequences mutually indistinguishable destroys precisely the
fine distinctions the task depends on. Loss still fell (v1: 1.489 ->
1.274) because collapsing coarse clusters is an easy way to reduce a
coarse-label contrastive loss -- a reminder that a falling loss is not
evidence the right thing is being learned.
  Fix: the class label is the INSTANCE, not its taxon. Exact-duplicate
  sequences (well documented in this database, Parts 3-4) share a label
  so they count as positives rather than false negatives.

Cause 3 -- TRAIN/TEST CONTAMINATION. v1 fine-tuned on the entire 5,000
record table, which includes the 1,000 records the evaluation holds out
to generate its test fragments. That inflates fine-tuned scores (so it
cannot explain the regression, but it does mean v1's fish and eukaryota
gains are partly unearned).
  Fix: training uses ONLY the train side of the identical seed-42 split
  the evaluation builds, imported from the evaluation's own split
  function, so the held-out references are genuinely unseen.

Reuses supervised_contrastive_loss, mean_pool and the batching/pooling
conventions from src/finetune_dnabert.py UNCHANGED -- the objective's
mathematics is not what was wrong, only what it was applied to.

Known, accepted noise source (stated rather than engineered around):
two references drawn into the same batch could be >=95% identical
without being byte-identical, in which case one is treated as a negative
for the other. With 8 instances sampled per batch from a 4,000-record
pool this is rare, and detecting it would require a pairwise alignment
per batch; left as a documented limitation.

Usage:
    python -m src.finetune_dnabert_fragment <kingdom> <out_model_dir> [epochs]
    kingdom in {bacteria, eukaryota, fish}
"""
import json
import os
import random
import sys
import time

import torch

# torch defaults to half this machine's cores; training is CPU-bound here.
torch.set_num_threads(os.cpu_count() or 8)

# Number of top transformer layers left trainable; the rest (and the embedding
# layer) are frozen. Freezing shrinks the backward pass, which is the dominant
# CPU cost -- and with only ~4,000 training references, updating all 117M
# parameters is over-parameterized anyway and risks destroying the pretrained
# representation, which is plausibly part of what went wrong in v1. Set to None
# to fine-tune the whole model.
TRAINABLE_TOP_LAYERS = 4

from .finetune_dnabert import (
    MODEL_NAME, MAX_LENGTH, LEARNING_RATE, TEMPERATURE, SEED,
    mean_pool, supervised_contrastive_loss,
)

FRAGMENT_LENGTHS = list(range(118, 141))  # the real read-length distribution used everywhere in this project
INSTANCES_PER_BATCH = 8                    # 8 instances x 2 views = 16 sequences per batch, same cost as v1
EPOCHS = 2

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


def get_train_sequences(kingdom: str):
    """Returns ONLY the train side of the exact split the evaluation uses,
    so the evaluation's held-out references are never seen in training."""
    if kingdom == "fish":
        from .stage0_diagnostics import build_holdout_split
        split = build_holdout_split()
    else:
        from .other_kingdom_diagnostics import build_holdout_split
        from .evaluate_finetuned import REF_TABLES
        ref_table_csv, ref_emb_npy, ref_ids_csv = REF_TABLES[kingdom]
        split = build_holdout_split(ref_table_csv, ref_emb_npy, ref_ids_csv)
    return split["train_ids"], split["train_seqs"]


def freeze_lower_layers(model, trainable_top_layers):
    """Freezes everything except the top N transformer layers. Returns
    (n_trainable_params, n_total_params). No-op if trainable_top_layers is None."""
    total = sum(p.numel() for p in model.parameters())
    if trainable_top_layers is None:
        return total, total

    layers = None
    for attr_path in (("encoder", "layer"), ("layers",), ("encoder", "layers")):
        obj = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break
    if layers is None:
        print("WARNING: could not locate transformer layers to freeze -- training all parameters")
        return total, total

    n_layers = len(layers)
    keep_from = max(0, n_layers - trainable_top_layers)
    for p in model.parameters():
        p.requires_grad = False
    for layer in layers[keep_from:]:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Froze layers 0-{keep_from - 1} of {n_layers}; training top {n_layers - keep_from} layers "
          f"({trainable / 1e6:.1f}M of {total / 1e6:.1f}M parameters)")
    return trainable, total


def make_views(seq: str, rng: random.Random):
    """One training pair: a random read-length fragment of the reference,
    and the full reference it came from."""
    length = rng.choice(FRAGMENT_LENGTHS)
    if len(seq) <= length:
        fragment = seq
    else:
        start = rng.randint(0, len(seq) - length)
        fragment = seq[start:start + length]
    return fragment, seq


def finetune(kingdom: str, out_model_dir: str, epochs: int = EPOCHS, seed: int = SEED):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_ids, train_seqs = get_train_sequences(kingdom)
    # instance label = the sequence itself, so exact duplicates are positives, not false negatives
    instance_label = {i: seq for i, seq in enumerate(train_seqs)}
    n_distinct = len(set(train_seqs))
    print(f"[{kingdom}] train references: {len(train_seqs)} ({n_distinct} distinct sequences)")
    print(f"[{kingdom}] held-out references are EXCLUDED from training (same seed-42 split as the evaluation)")

    _load_model()
    _model.train()
    n_trainable, n_total = freeze_lower_layers(_model, TRAINABLE_TOP_LAYERS)
    opt = torch.optim.AdamW([p for p in _model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    steps_per_epoch = max(1, len(train_seqs) // INSTANCES_PER_BATCH)
    print(f"[{kingdom}] {epochs} epochs x {steps_per_epoch} steps/epoch, "
          f"{INSTANCES_PER_BATCH} instances x 2 views = {INSTANCES_PER_BATCH * 2} sequences/batch")

    order = list(range(len(train_seqs)))
    loss_log = []
    t0 = time.time()

    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_losses = []
        for step in range(steps_per_epoch):
            picks = order[step * INSTANCES_PER_BATCH:(step + 1) * INSTANCES_PER_BATCH]
            if not picks:
                continue

            batch_seqs, batch_labels = [], []
            for idx in picks:
                fragment, full = make_views(train_seqs[idx], rng)
                batch_seqs.extend([fragment, full])
                batch_labels.extend([instance_label[idx], instance_label[idx]])

            enc = _tokenizer(batch_seqs, padding=True, truncation=True,
                             max_length=MAX_LENGTH, return_tensors="pt")
            out = _model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=1)

            loss = supervised_contrastive_loss(pooled, batch_labels, TEMPERATURE)
            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_losses.append(float(loss.item()))
            if (step + 1) % 50 == 0:
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
        "kingdom": kingdom,
        "objective": "instance-level contrastive, fragment<->full-amplicon views",
        "fixes_vs_v1": [
            "trains on read-length fragments paired with their own reference (input mismatch)",
            "instance-level labels instead of coarse taxonomy classes (label granularity)",
            "trains only on the evaluation's train split (train/test contamination)",
        ],
        "n_train_references": len(train_seqs),
        "n_distinct_sequences": n_distinct,
        "trainable_top_layers": TRAINABLE_TOP_LAYERS,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "torch_threads": torch.get_num_threads(),
        "fragment_lengths": [FRAGMENT_LENGTHS[0], FRAGMENT_LENGTHS[-1]],
        "instances_per_batch": INSTANCES_PER_BATCH,
        "views_per_instance": 2,
        "epochs": epochs,
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
    kingdom_arg, out_dir = sys.argv[1], sys.argv[2]
    n_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS
    finetune(kingdom_arg, out_dir, epochs=n_epochs)

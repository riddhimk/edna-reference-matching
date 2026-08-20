"""
Shared DNABERT-S embedding logic used by BOTH reference and experimental
embedding stages, so the two embedding spaces stay comparable.

Model/tokenizer are loaded lazily (only when embed_batch() is first called),
so importing this module doesn't require network access or a GPU -- this
keeps it safe to import from tests and from other stages that don't embed.
"""
from typing import List, Optional
import numpy as np

from . import config
from .fasta_io import to_dna

_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    import torch
    import multimolecule  # noqa: F401 -- registers multimolecule/* architectures with transformers Auto* classes
    from transformers import AutoTokenizer, AutoModel

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(config.MODEL_NAME, trust_remote_code=True).to(_device).eval()

def embed_batch(seqs: List[str], max_length: Optional[int] = None) -> np.ndarray:
    """Embed a batch of RNA or DNA sequences with DNABERT-S.

    Applies U->T conversion (SILVA sequences are RNA). Uses dynamic padding
    at batch time -- never edits the stored sequence to force equal length.
    Pooling strategy (mean vs CLS) is fixed by config.POOLING and MUST match
    whatever was used to produce your existing experimental-read embeddings.
    """
    import torch

    _load_model()
    max_length = max_length or config.MAX_LENGTH
    if max_length is None:
        raise ValueError(
            "max_length is not set. Do not guess this (e.g. defaulting to 140) -- "
            "set config.MAX_LENGTH from inspect_silva.py's percentile output AND "
            "after resolving the region-mismatch question (see README.md)."
        )

    dna_seqs = [to_dna(s) for s in seqs]
    enc = _tokenizer(
        dna_seqs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        out = _model(**enc)
    hidden = out.last_hidden_state                # (batch, seq_len, 768)
    mask = enc["attention_mask"].unsqueeze(-1)     # (batch, seq_len, 1)

    if config.POOLING == "mean":
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
    elif config.POOLING == "cls":
        pooled = hidden[:, 0, :]
    else:
        raise ValueError(f"Unknown POOLING config: {config.POOLING!r} (expected 'mean' or 'cls')")

    return pooled.cpu().numpy().astype("float32")

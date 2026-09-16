"""
GENA-LM (bert-base-t2t) embedding -- fourth open-source model touchpoint
in the model-vs-model comparison, alongside src/nt_embed.py (Nucleotide
Transformer) and src/hyenadna_embed.py (HyenaDNA). Kept fully separate
from src/embed_common.py and src/config.py on purpose: nothing here is
imported by embed_common.py, final_pipeline.py, or any other existing
stage, so the DNABERT-S pipeline and every frozen Part 3/4 artifact stay
untouched.

Model note: a standard BERT architecture (AIRI-Institute/GENA-LM,
published in NAR Genomics and Bioinformatics), trained on the T2T human
reference assembly -- a different pretraining corpus/objective from both
DNABERT-S (species-contrastive) and Nucleotide Transformer (multi-species
masked-LM). Confirmed via direct source inspection (modeling_bert.py) to
use plain PyTorch/transformers only -- no triton/flash-attn/custom CUDA
kernel -- before running anything at scale. (DNABERT-2 was also
considered and rejected here: it loads, but its remote code calls a
Triton flash-attention kernel that hard-requires an actual CUDA GPU even
when triton itself is importable -- confirmed to crash on this CPU-only
machine, not assumed.)
"""
from typing import List, Optional
import numpy as np

MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t"
EMBEDDING_DIM = 768
MAX_LENGTH = 256  # tokens; comfortably covers MitoFish's <250bp amplicons

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    from transformers import AutoTokenizer, AutoModel

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()


def embed_batch(seqs: List[str], max_length: Optional[int] = None) -> np.ndarray:
    """Mean-pools the final hidden layer over non-padding tokens -- same
    pooling strategy as embed_common.py's DNABERT path and nt_embed.py,
    so all models are compared on equal footing. Input sequences are
    expected already in DNA form (T, not U)."""
    import torch

    _load_model()
    max_length = max_length or MAX_LENGTH
    enc = _tokenizer(seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        out = _model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    mask = enc["attention_mask"].unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled.numpy().astype("float32")

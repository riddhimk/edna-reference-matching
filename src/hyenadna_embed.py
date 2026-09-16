"""
HyenaDNA-tiny (1k seqlen) embedding -- third open-source model in the
model-vs-model comparison (see src/nt_embed.py for the first, and
src/genalm_embed.py for the second). Kept fully separate from
src/embed_common.py and src/config.py on purpose: nothing here is
imported by embed_common.py, final_pipeline.py, or any other existing
stage, so the DNABERT-S pipeline and every frozen Part 3/4 artifact stay
untouched.

Architecture note: HyenaDNA uses implicit long convolutions (a
state-space/Hyena operator), not attention -- a genuinely different
architecture family from DNABERT-S/Nucleotide Transformer's transformer
attention, chosen deliberately for a diverse comparison rather than
picking three variants of the same design. Confirmed via direct source
inspection (modeling_hyena.py) to use plain PyTorch only -- no
triton/flash-attn/custom CUDA kernel -- before running anything at scale.
"""
from typing import List, Optional
import numpy as np

MODEL_NAME = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
EMBEDDING_DIM = 128
MAX_LENGTH = 256  # tokens; HyenaDNA is character-level, comfortably covers MitoFish's <250bp amplicons

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
    expected already in DNA form (T, not U).

    HyenaDNA's tokenizer returns no attention_mask (it's a char-level
    causal-LM tokenizer, not built for masked/bidirectional pooling), so
    the mask is built here from input_ids != pad_token_id instead."""
    import torch

    _load_model()
    max_length = max_length or MAX_LENGTH
    enc = _tokenizer(seqs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        out = _model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    mask = (enc["input_ids"] != _tokenizer.pad_token_id).unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled.numpy().astype("float32")

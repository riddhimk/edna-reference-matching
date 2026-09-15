"""
Nucleotide Transformer v2 (50M, multi-species) embedding -- a second,
independent open-source model run alongside the existing DNABERT-S/
multimolecule pipeline (src/embed_common.py), for a real model-vs-model
comparison (professor's "open-source LLM models for retrieval" /
"comparison of models" next steps).

Kept fully separate from src/embed_common.py and src/config.py on
purpose: nothing here is imported by embed_common.py, final_pipeline.py,
or any other existing stage, so the DNABERT-S pipeline and every frozen
Part 3/4 artifact stay untouched.

Model choice rationale (see chat record / README roadmap): standard
transformers architecture, no triton/flash-attn/custom-kernel dependency
-- the exact thing that blocked zhihan1996/DNABERT-S on this Windows/CPU
setup. Confirmed to load and run on this machine's CPU before any
large-scale embedding was run.
"""
from typing import List, Optional
import numpy as np

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
EMBEDDING_DIM = 512
MAX_LENGTH = 256  # tokens; NT's 6-mer tokenizer, comfortably covers MitoFish's <250bp amplicons

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()


def embed_batch(seqs: List[str], max_length: Optional[int] = None) -> np.ndarray:
    """Mean-pools the final hidden layer over non-padding tokens -- same
    pooling strategy as embed_common.py's DNABERT path, so the two models
    are compared on equal footing. Input sequences are expected already in
    DNA form (T, not U) -- MitoFish's extracted amplicons, not SILVA, so no
    U->T conversion is needed here (unlike embed_common.py)."""
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

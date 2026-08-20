"""
Central configuration for the SILVA reference-matching stage.

Nothing here is a validated result — MAX_LENGTH, SIMILARITY_THRESHOLD and
MIN_MARGIN are placeholders you MUST set from real analysis before trusting
the pipeline's output:
    - MAX_LENGTH            -> set from the percentile output of inspect_silva.py
                               AND from resolving the region-mismatch question
                               (see README.md, "Region mismatch").
    - SIMILARITY_THRESHOLD  -> set from the labeled-holdout sweep in evaluate.py.
    - MIN_MARGIN            -> same as above.
Ship this file with placeholders = None/obviously-fake so nobody accidentally
treats an unset value as validated.
"""
import os

# ---- paths (override via env vars so nothing is hardcoded per-machine) ----
# NOTE: these defaults point at data/SILVA results/ -- all SILVA-specific
# outputs were moved there to keep them separate from the MitoFish
# reference-matching run (data/MitoFish results/). Every actual pipeline
# invocation in this project's README/Explanation.md passes these paths
# explicitly via env vars anyway; these defaults just mean `python
# run_pipeline.py <stage>` with no env override still works out of the box
# for the SILVA pipeline specifically.
SILVA_FASTA_PATH = os.environ.get(
    "SILVA_FASTA_PATH", "data/SILVA results/SILVA_138.2_SSURef_tax_silva_trunc.fasta"
)
REFERENCE_TABLE_CSV = os.environ.get("REFERENCE_TABLE_CSV", "data/SILVA results/silva_reference_table.csv")
REFERENCE_EMBEDDINGS_NPY = os.environ.get("REFERENCE_EMBEDDINGS_NPY", "data/SILVA results/silva_ref_embeddings.npy")
REFERENCE_IDS_CSV = os.environ.get("REFERENCE_IDS_CSV", "data/SILVA results/silva_ref_ids.csv")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/SILVA results/silva_faiss.index")

EXPERIMENTAL_FASTQ_PATH = os.environ.get("EXPERIMENTAL_FASTQ_PATH", "data/ERR16922870_1.fastq")
EXPERIMENTAL_EMBEDDINGS_NPY = os.environ.get("EXPERIMENTAL_EMBEDDINGS_NPY", "data/reads_embeddings.npy")

# Read_ID -> Cluster_ID from your EXISTING HDBSCAN stage (18 clusters, 218 noise).
# This file is produced by your clustering code, not by this repo -- just point
# this at it. Expected columns: Read_ID, Cluster_ID (HDBSCAN noise is usually -1).
CLUSTER_ASSIGNMENTS_CSV = os.environ.get("CLUSTER_ASSIGNMENTS_CSV", "data/hdbscan_cluster_assignments.csv")
CLUSTER_ANNOTATION_CSV = os.environ.get("CLUSTER_ANNOTATION_CSV", "data/cluster_read_annotations.csv")
CLUSTER_SUMMARY_CSV = os.environ.get("CLUSTER_SUMMARY_CSV", "data/cluster_summary.csv")

# How many leading taxonomy fields to join as the pseudo-label used for cluster
# purity (e.g. depth=3 -> "Eukaryota;Amorphea;Obazoa"). Because SILVA taxonomy
# depth is NOT fixed (8 fields vs 18 fields seen in the two real records
# inspected), a deeper value gives finer-grained but noisier labels, and a
# shallow value gives coarser but more stable labels. There is no single
# correct value -- try a couple and report which you used.
CLUSTER_LABEL_TAXONOMY_DEPTH = 3

RESULTS_CSV = os.environ.get("RESULTS_CSV", "data/SILVA results/read_taxonomy_results.csv")

# ---- model ----
# zhihan1996/DNABERT-S's remote code hard-requires `triton`, which has no
# Windows/CPU build -- confirmed a hard blocker on this machine, not a
# preference. Using multimolecule/dnaberts instead (also named in the
# project PDF as the model used for the original read embeddings); it loads
# on CPU with no triton dependency and produces the same 768-dim output.
MODEL_NAME = "multimolecule/dnaberts"
EMBEDDING_DIM = 768

# POOLING must match whatever pooling was used to produce your existing 5,000
# read embeddings. If you don't already know, check that code before trusting
# any similarity number this pipeline produces.
POOLING = "mean"  # "mean" or "cls"

BATCH_SIZE = 64

# MAX_LENGTH is a TOKEN count (passed straight to the tokenizer's
# max_length=), not a bp count. Derived from real Stage 1 inspection output
# on the full SILVA file (2,224,690 records): p95 reference length = 1696bp,
# mean = 1427bp. Empirically measured multimolecule/dnaberts tokenization
# ratio on real SILVA sequences: ~0.20-0.22 tokens/bp (e.g. 1860bp -> 404
# tokens). 450 tokens therefore covers the large majority of full-length
# references without truncation. Experimental reads (118-140bp, ~25-30
# tokens) are never truncated by this value -- so this setting is driven by
# the reference side only.
# NOT a fix for the region-mismatch limitation documented in README.md: the
# references are near-full-length SSU rRNA (900-3718bp) while the reads are
# a short 118-140bp fragment of unknown position within that gene. Both are
# embedded whole (no alignment/region-extraction), so a read's mean-pooled
# vector is compared against a full-length reference's mean-pooled vector.
# This is a real, documented scope limitation of this POC, not a bug fixed
# by this constant.
MAX_LENGTH = 450

# ---- FAISS / matching ----
TOP_K = 5  # retrieve top-K so margin-to-runner-up can be computed

# Derived from src/synthetic_read_calibration.py, run on a real held-out
# split of the SILVA reference sample (1,000 synthetic 118-140bp fragments
# cut at a RANDOM position from held-out full-length references, matching
# the real ERR16922870 read-length distribution, queried against the other
# 4,000 full-length references).
#
# Finding: accuracy stayed at ~0.09-0.10 (near chance, taxonomy_depth=3)
# across the ENTIRE similarity range tested (0.10-0.68) -- there is no
# threshold at which a random-position short fragment reliably matches its
# true parent's taxonomy against full-length references. This validates
# README.md's "Region mismatch" warning empirically: without resolving
# which primer/hypervariable region produced the real reads, short-fragment
# vs. full-length-reference matching is not diagnostic in this embedding
# space, regardless of where the threshold is set.
# 0.60 is set near the max similarity observed in BOTH the real read query
# results (max 0.597) and this calibration (max 0.620) specifically so that
# Stage E conservatively routes reads to "unclassified" -> HDBSCAN open-set
# (Capability 2) rather than forcing an unsupported taxonomic call -- this
# is the correct behavior given the calibration evidence, not a placeholder.
SIMILARITY_THRESHOLD = 0.60
MIN_MARGIN = None

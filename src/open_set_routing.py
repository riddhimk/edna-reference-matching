"""
Stage 11 -- Split reads into (assigned, unclassified) and route unclassified
reads' EXISTING embeddings into the already-completed HDBSCAN stage.

Architecture is preserved as-is: this module does not touch or re-run
HDBSCAN/UMAP -- it only decides which reads feed into them.

IMPORTANT DISTINCTION (see README.md):
"low-confidence reference match" != "new species". A low similarity score
can come from sequencing error, a short/low-quality read, wrong gene region,
a reference-database gap, or an embedding artifact -- not just novelty.
Label output "unclassified" / "below reference-match confidence", never
"novel taxon", until independent clustering evidence supports that claim.
"""
from typing import Dict, List, Tuple


def split_for_open_set(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    assigned = [r for r in results if r["Confidence_Decision"] == "assigned"]
    unclassified = [r for r in results if r["Confidence_Decision"] == "unclassified"]
    return assigned, unclassified


def get_unclassified_read_ids(results: List[Dict]) -> List[str]:
    """Read_IDs to feed into the existing HDBSCAN clustering stage.

    Use these IDs to look up the reads' ALREADY-COMPUTED 768-D embeddings
    from your existing pipeline -- do not re-embed them here.
    """
    _, unclassified = split_for_open_set(results)
    return [r["Read_ID"] for r in unclassified]

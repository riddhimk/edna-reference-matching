"""
Stage 9-10 -- Attach SILVA taxonomy to each read's top hit and decide
assigned vs. unclassified.

CRITICAL: threshold and min_margin are NOT hardcoded here. Passing None
(the config default) makes decide() refuse to run rather than silently
using an unvalidated number like 0.8 or 0.9. Derive real values from
evaluate.py's threshold sweep against a labeled holdout set first.
"""
import csv
from typing import Dict, List, Optional

from . import config


def load_ref_lookup(ref_ids_csv: str, ref_table_csv: str) -> Dict[int, Dict]:
    """Build {row_index -> {Reference_ID, Taxonomy_Full}} for attaching taxonomy
    to a FAISS index position returned by query_index.py."""
    pos_to_id = {}
    with open(ref_ids_csv) as f:
        for row in csv.DictReader(f):
            pos_to_id[int(row["row_index"])] = row["Reference_ID"]

    id_to_taxonomy = {}
    with open(ref_table_csv) as f:
        for row in csv.DictReader(f):
            id_to_taxonomy[row["Reference_ID"]] = row["Taxonomy_Full"]

    lookup = {}
    for pos, ref_id in pos_to_id.items():
        lookup[pos] = {"Reference_ID": ref_id, "Taxonomy_Full": id_to_taxonomy.get(ref_id, "")}
    return lookup


def decide(similarity: float, margin_to_2nd: float, threshold: float,
           min_margin: Optional[float] = None) -> str:
    """Return 'assigned' or 'unclassified'.

    threshold / min_margin must be experimentally-derived values passed in
    explicitly -- see evaluate.py. This function does not choose them.
    """
    if threshold is None:
        raise ValueError(
            "threshold is None. Do not guess a value (e.g. 0.8/0.9) -- derive it "
            "from evaluate.py's precision/recall sweep on a labeled holdout set."
        )
    if similarity < threshold:
        return "unclassified"
    if min_margin is not None and margin_to_2nd < min_margin:
        return "unclassified"  # ambiguous -- too close to the runner-up to trust
    return "assigned"


def build_results(query_results_csv: str, ref_ids_csv: str, ref_table_csv: str,
                   threshold: float, min_margin: Optional[float], out_csv: str) -> List[Dict]:
    lookup = load_ref_lookup(ref_ids_csv, ref_table_csv)
    results = []
    with open(query_results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            top1_pos = int(row["top1_index_pos"])
            top1_sim = float(row["top1_similarity"])
            top2_sim = float(row.get("top2_similarity", top1_sim))
            margin = top1_sim - top2_sim
            ref = lookup.get(top1_pos, {"Reference_ID": "", "Taxonomy_Full": ""})
            decision = decide(top1_sim, margin, threshold, min_margin)
            results.append({
                "Read_ID": row["Read_ID"],
                "Nearest_Reference_ID": ref["Reference_ID"],
                "Similarity": top1_sim,
                "Margin_to_2nd": margin,
                "Taxonomy": ref["Taxonomy_Full"],
                "Confidence_Decision": decision,
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Read_ID", "Nearest_Reference_ID", "Similarity", "Margin_to_2nd",
            "Taxonomy", "Confidence_Decision",
        ])
        w.writeheader()
        w.writerows(results)

    n_assigned = sum(1 for r in results if r["Confidence_Decision"] == "assigned")
    print(f"{n_assigned}/{len(results)} reads assigned at threshold={threshold}, min_margin={min_margin}")
    return results


if __name__ == "__main__":
    if config.SIMILARITY_THRESHOLD is None:
        raise SystemExit(
            "config.SIMILARITY_THRESHOLD is not set. Run evaluate.py's threshold "
            "sweep on a labeled holdout set first, then set it in src/config.py."
        )
    build_results(
        query_results_csv="data/query_results.csv",
        ref_ids_csv=config.REFERENCE_IDS_CSV,
        ref_table_csv=config.REFERENCE_TABLE_CSV,
        threshold=config.SIMILARITY_THRESHOLD,
        min_margin=config.MIN_MARGIN,
        out_csv=config.RESULTS_CSV,
    )

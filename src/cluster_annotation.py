"""
Cluster validation against the SILVA reference table.

Your HDBSCAN stage is already done (18 clusters, 218 noise) and does NOT get
re-run or changed by this module. What this module does:

    for every read (regardless of which cluster it's in, including noise):
        find its nearest SILVA reference via FAISS (already computed in
        data/query_results.csv from query_index.py)
        attach that reference's taxonomy as a PSEUDO-LABEL for the read

    for every cluster:
        look at the pseudo-labels of its member reads
        report whether they agree (purity) or are scattered (impure/mixed)

This answers "do our density-based clusters correspond to real taxonomic
groups, according to a standard reference database" -- which is the
validation your professor is asking for.

IMPORTANT FRAMING (do not skip this when writing it up):
The SILVA nearest-neighbor hit is a PSEUDO-label, not verified ground truth.
A read's top-1 FAISS match can be wrong (short read, reference-database gap,
sequencing error) independent of whether HDBSCAN's clustering is good.
High cluster purity is evidence the clusters are taxonomically coherent, not
proof every individual assignment is correct. Report it as "agreement with
SILVA nearest-reference matches", not as "% accuracy".
"""
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from . import config


def taxonomy_prefix(taxonomy_full: str, depth: int) -> str:
    """Join the first `depth` semicolon-delimited taxonomy fields.

    Used as the pseudo-label for cluster purity. Because SILVA taxonomy
    depth is variable (8 fields vs 18 fields seen in the two real records
    inspected -- see fasta_io.py), a record with fewer than `depth` fields
    just uses however many it has, rather than erroring or padding.
    """
    if not taxonomy_full:
        return ""
    fields = [f.strip() for f in taxonomy_full.split(";") if f.strip()]
    return ";".join(fields[:depth])


def load_query_results(query_results_csv: str) -> Dict[str, Dict]:
    """Read_ID -> {top1_index_pos, top1_similarity} from query_index.py output."""
    out = {}
    with open(query_results_csv) as f:
        for row in csv.DictReader(f):
            out[row["Read_ID"]] = {
                "top1_index_pos": int(row["top1_index_pos"]),
                "top1_similarity": float(row["top1_similarity"]),
            }
    return out


def load_cluster_assignments(csv_path: str) -> Dict[str, str]:
    """Read_ID -> Cluster_ID, from your existing HDBSCAN stage's output."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["Read_ID"]] = row["Cluster_ID"]
    return out


def build_read_labels(query_results: Dict[str, Dict], ref_lookup: Dict[int, Dict],
                       taxonomy_depth: int, min_similarity: Optional[float] = None) -> Dict[str, Dict]:
    """Attach a SILVA pseudo-label to each read that has a FAISS query result.

    If min_similarity is given, reads below it are labeled with
    Taxonomy_Label = "" (no confident reference match) rather than being
    dropped -- they still count toward cluster size, just not toward purity.
    """
    read_labels = {}
    for read_id, q in query_results.items():
        ref = ref_lookup.get(q["top1_index_pos"], {"Reference_ID": "", "Taxonomy_Full": ""})
        sim = q["top1_similarity"]
        has_confident_match = (min_similarity is None) or (sim >= min_similarity)
        label = taxonomy_prefix(ref["Taxonomy_Full"], taxonomy_depth) if has_confident_match else ""
        read_labels[read_id] = {
            "Nearest_Reference_ID": ref["Reference_ID"],
            "Similarity": sim,
            "Taxonomy_Full": ref["Taxonomy_Full"],
            "Taxonomy_Label": label,
            "Has_Reference_Match": has_confident_match,
        }
    return read_labels


def annotate_clusters(cluster_assignments: Dict[str, str], read_labels: Dict[str, Dict],
                       noise_label: str = "-1") -> List[Dict]:
    """Per-cluster purity/coherence summary.

    Purity = (count of the most common Taxonomy_Label in the cluster) /
             (count of reads in the cluster that HAD a reference match).
    Noise-labeled reads (HDBSCAN's -1, or whatever noise_label you pass) are
    reported as their own row, not folded into a real cluster's purity.
    """
    clusters: Dict[str, List[str]] = defaultdict(list)
    for read_id, cluster_id in cluster_assignments.items():
        clusters[cluster_id].append(read_id)

    summary = []
    for cluster_id, read_ids in sorted(clusters.items(), key=lambda kv: (kv[0] == noise_label, kv[0])):
        n_reads = len(read_ids)
        labels_in_cluster = [
            read_labels[rid]["Taxonomy_Label"]
            for rid in read_ids
            if rid in read_labels and read_labels[rid]["Has_Reference_Match"] and read_labels[rid]["Taxonomy_Label"]
        ]
        n_with_match = len(labels_in_cluster)
        n_no_match_or_missing = n_reads - n_with_match

        if labels_in_cluster:
            counts = Counter(labels_in_cluster)
            dominant_label, dominant_count = counts.most_common(1)[0]
            purity = dominant_count / n_with_match
            n_distinct_taxa = len(counts)
        else:
            dominant_label, dominant_count, purity, n_distinct_taxa = "", 0, None, 0

        summary.append({
            "Cluster_ID": cluster_id,
            "Is_Noise": cluster_id == noise_label,
            "N_Reads": n_reads,
            "N_With_Reference_Match": n_with_match,
            "Pct_With_Reference_Match": round(n_with_match / n_reads, 4) if n_reads else None,
            "Dominant_Taxonomy_Label": dominant_label,
            "Dominant_Label_Count": dominant_count,
            "Purity": round(purity, 4) if purity is not None else None,
            "N_Distinct_Taxa_In_Cluster": n_distinct_taxa,
        })
    return summary


def compute_agreement_metrics(cluster_assignments: Dict[str, str], read_labels: Dict[str, Dict],
                               noise_label: str = "-1") -> Dict:
    """Global agreement between HDBSCAN clusters and SILVA pseudo-labels,
    using standard clustering-vs-labels metrics (the ARI/NMI/homogeneity/
    completeness already planned for this project's evaluation).

    Only reads that (a) are not noise and (b) have a confident reference
    match are included -- this is stated explicitly in the returned dict
    so it's never silently mixed up with "evaluated on all 5,000 reads".
    """
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        homogeneity_score, completeness_score, v_measure_score,
    )

    read_ids = [
        rid for rid in cluster_assignments
        if cluster_assignments[rid] != noise_label
        and rid in read_labels
        and read_labels[rid]["Has_Reference_Match"]
        and read_labels[rid]["Taxonomy_Label"]
    ]
    n_used = len(read_ids)
    n_total = len(cluster_assignments)

    if n_used < 2:
        return {"n_reads_used": n_used, "n_reads_total": n_total, "note": "too few labeled non-noise reads to compute metrics"}

    cluster_ids = [cluster_assignments[rid] for rid in read_ids]
    taxonomy_labels = [read_labels[rid]["Taxonomy_Label"] for rid in read_ids]

    return {
        "n_reads_used": n_used,
        "n_reads_total": n_total,
        "pct_of_all_reads_used": round(n_used / n_total, 4) if n_total else None,
        "adjusted_rand_index": adjusted_rand_score(taxonomy_labels, cluster_ids),
        "normalized_mutual_info": normalized_mutual_info_score(taxonomy_labels, cluster_ids),
        "homogeneity": homogeneity_score(taxonomy_labels, cluster_ids),
        "completeness": completeness_score(taxonomy_labels, cluster_ids),
        "v_measure": v_measure_score(taxonomy_labels, cluster_ids),
        "note": "excludes HDBSCAN noise reads and reads with no confident SILVA match",
    }


def write_cluster_annotation(cluster_assignments: Dict[str, str], read_labels: Dict[str, Dict], out_csv: str):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Read_ID", "Cluster_ID", "Nearest_Reference_ID", "Similarity",
            "Taxonomy_Label", "Has_Reference_Match",
        ])
        w.writeheader()
        for rid, cluster_id in cluster_assignments.items():
            rl = read_labels.get(rid, {})
            w.writerow({
                "Read_ID": rid,
                "Cluster_ID": cluster_id,
                "Nearest_Reference_ID": rl.get("Nearest_Reference_ID", ""),
                "Similarity": rl.get("Similarity", ""),
                "Taxonomy_Label": rl.get("Taxonomy_Label", ""),
                "Has_Reference_Match": rl.get("Has_Reference_Match", False),
            })


def write_cluster_summary(summary: List[Dict], out_csv: str):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Cluster_ID", "Is_Noise", "N_Reads", "N_With_Reference_Match",
            "Pct_With_Reference_Match", "Dominant_Taxonomy_Label",
            "Dominant_Label_Count", "Purity", "N_Distinct_Taxa_In_Cluster",
        ])
        w.writeheader()
        w.writerows(summary)


def print_cluster_summary(summary: List[Dict]) -> None:
    print(f"{'Cluster':>8} {'N_reads':>8} {'%matched':>9} {'purity':>8} {'#taxa':>6}  dominant taxon")
    for row in summary:
        pct = f"{row['Pct_With_Reference_Match']*100:.1f}%" if row["Pct_With_Reference_Match"] is not None else "NA"
        purity = f"{row['Purity']:.3f}" if row["Purity"] is not None else "NA"
        tag = " (noise)" if row["Is_Noise"] else ""
        print(f"{row['Cluster_ID']:>8}{tag} {row['N_Reads']:>8} {pct:>9} {purity:>8} {row['N_Distinct_Taxa_In_Cluster']:>6}  {row['Dominant_Taxonomy_Label']}")

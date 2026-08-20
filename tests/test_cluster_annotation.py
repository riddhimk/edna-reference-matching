from src.cluster_annotation import (
    taxonomy_prefix, build_read_labels, annotate_clusters, compute_agreement_metrics,
)

# Mirrors the real inspected records' shape: variable depth taxonomy strings
TAX_A = "Eukaryota;Archaeplastida;Chloroplastida;Chlorophyta;Chlorophyceae"
TAX_B = "Eukaryota;Amorphea;Obazoa;Opisthokonta;Holozoa;Choanozoa;Metazoa"


def test_taxonomy_prefix_basic():
    assert taxonomy_prefix(TAX_A, depth=3) == "Eukaryota;Archaeplastida;Chloroplastida"


def test_taxonomy_prefix_handles_shallow_string_gracefully():
    # only 2 real fields but depth=5 requested -- should not error or pad
    assert taxonomy_prefix("Eukaryota;Amorphea", depth=5) == "Eukaryota;Amorphea"


def test_taxonomy_prefix_empty_string():
    assert taxonomy_prefix("", depth=3) == ""


def test_build_read_labels_no_threshold_labels_everything():
    query_results = {"r1": {"top1_index_pos": 0, "top1_similarity": 0.9}}
    ref_lookup = {0: {"Reference_ID": "REF1", "Taxonomy_Full": TAX_A}}
    labels = build_read_labels(query_results, ref_lookup, taxonomy_depth=3, min_similarity=None)
    assert labels["r1"]["Has_Reference_Match"] is True
    assert labels["r1"]["Taxonomy_Label"] == "Eukaryota;Archaeplastida;Chloroplastida"


def test_build_read_labels_below_threshold_has_no_label():
    query_results = {"r1": {"top1_index_pos": 0, "top1_similarity": 0.4}}
    ref_lookup = {0: {"Reference_ID": "REF1", "Taxonomy_Full": TAX_A}}
    labels = build_read_labels(query_results, ref_lookup, taxonomy_depth=3, min_similarity=0.7)
    assert labels["r1"]["Has_Reference_Match"] is False
    assert labels["r1"]["Taxonomy_Label"] == ""


def test_annotate_clusters_pure_cluster():
    # cluster "0": 3 reads, all matching the same taxon -> purity 1.0
    cluster_assignments = {"r1": "0", "r2": "0", "r3": "0"}
    read_labels = {
        "r1": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r2": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r3": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
    }
    summary = annotate_clusters(cluster_assignments, read_labels)
    assert len(summary) == 1
    assert summary[0]["Purity"] == 1.0
    assert summary[0]["N_Distinct_Taxa_In_Cluster"] == 1


def test_annotate_clusters_mixed_cluster():
    # cluster "0": 4 reads, 3 agree, 1 disagrees -> purity 0.75
    cluster_assignments = {"r1": "0", "r2": "0", "r3": "0", "r4": "0"}
    read_labels = {
        "r1": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r2": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r3": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r4": {"Taxonomy_Label": "Eukaryota;X;Y", "Has_Reference_Match": True},
    }
    summary = annotate_clusters(cluster_assignments, read_labels)
    assert summary[0]["Purity"] == 0.75
    assert summary[0]["N_Distinct_Taxa_In_Cluster"] == 2


def test_annotate_clusters_noise_reported_separately():
    cluster_assignments = {"r1": "0", "r2": "-1"}
    read_labels = {
        "r1": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r2": {"Taxonomy_Label": "Eukaryota;X;Y", "Has_Reference_Match": True},
    }
    summary = annotate_clusters(cluster_assignments, read_labels)
    by_id = {row["Cluster_ID"]: row for row in summary}
    assert by_id["-1"]["Is_Noise"] is True
    assert by_id["0"]["Is_Noise"] is False


def test_annotate_clusters_unmatched_reads_excluded_from_purity_but_counted_in_size():
    cluster_assignments = {"r1": "0", "r2": "0"}
    read_labels = {
        "r1": {"Taxonomy_Label": "Eukaryota;A;B", "Has_Reference_Match": True},
        "r2": {"Taxonomy_Label": "", "Has_Reference_Match": False},
    }
    summary = annotate_clusters(cluster_assignments, read_labels)
    row = summary[0]
    assert row["N_Reads"] == 2
    assert row["N_With_Reference_Match"] == 1
    assert row["Purity"] == 1.0  # sole matched read is 100% of the matched subset


def test_compute_agreement_metrics_perfect_agreement():
    # clusters perfectly separate the two taxa -> ARI/homogeneity should be 1.0
    cluster_assignments = {"r1": "0", "r2": "0", "r3": "1", "r4": "1"}
    read_labels = {
        "r1": {"Taxonomy_Label": "A", "Has_Reference_Match": True},
        "r2": {"Taxonomy_Label": "A", "Has_Reference_Match": True},
        "r3": {"Taxonomy_Label": "B", "Has_Reference_Match": True},
        "r4": {"Taxonomy_Label": "B", "Has_Reference_Match": True},
    }
    metrics = compute_agreement_metrics(cluster_assignments, read_labels)
    assert metrics["adjusted_rand_index"] == 1.0
    assert metrics["homogeneity"] == 1.0
    assert metrics["n_reads_used"] == 4


def test_compute_agreement_metrics_excludes_noise_and_unmatched():
    cluster_assignments = {"r1": "0", "r2": "-1", "r3": "0"}
    read_labels = {
        "r1": {"Taxonomy_Label": "A", "Has_Reference_Match": True},
        "r2": {"Taxonomy_Label": "B", "Has_Reference_Match": True},   # noise -- excluded
        "r3": {"Taxonomy_Label": "", "Has_Reference_Match": False},   # unmatched -- excluded
    }
    metrics = compute_agreement_metrics(cluster_assignments, read_labels)
    assert metrics["n_reads_used"] == 1
    assert metrics["n_reads_total"] == 3


def test_compute_agreement_metrics_too_few_reads_returns_note():
    metrics = compute_agreement_metrics({"r1": "0"}, {"r1": {"Taxonomy_Label": "A", "Has_Reference_Match": True}})
    assert "note" in metrics
    assert metrics["n_reads_used"] < 2

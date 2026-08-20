"""
Stage 12 -- Evaluate the reference-matching stage against a labeled holdout
set, and sweep the similarity threshold to pick a real, justified operating
point instead of guessing 0.8/0.9.

No numbers in this module are pre-filled results -- everything here is
computed from whatever labeled data you pass in.

Usage pattern:
    y_true, y_pred, similarities = ...  # from your holdout run
    sweep = sweep_thresholds(y_true, y_pred, similarities, thresholds=[0.5,0.6,0.7,0.8,0.9])
    metrics = evaluate(y_true, y_pred, labels=sorted(set(y_true)))
"""
from typing import List, Sequence, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate(y_true_rank: Sequence[str], y_pred_rank: Sequence[str], labels: Sequence[str]) -> Dict:
    """Rank-level classification metrics (e.g. at Genus). No fabricated numbers --
    everything is computed from the y_true/y_pred you supply."""
    acc = accuracy_score(y_true_rank, y_pred_rank)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_rank, y_pred_rank, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true_rank, y_pred_rank, labels=labels)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "labels": list(labels),
    }


def sweep_thresholds(y_true_rank: Sequence[str], y_pred_rank: Sequence[str],
                      similarities: Sequence[float], thresholds: List[float]) -> List[Dict]:
    """For each candidate threshold, report what fraction of the holdout set
    would be 'assigned' at that threshold, and accuracy/precision/recall
    computed ONLY over the assigned subset. Use this curve to pick a
    threshold with a stated, defensible trade-off -- not a single cherry-picked
    number.
    """
    rows = []
    for t in thresholds:
        idx = [i for i, s in enumerate(similarities) if s >= t]
        pct_assigned = len(idx) / len(similarities) if similarities else 0.0
        if idx:
            yt = [y_true_rank[i] for i in idx]
            yp = [y_pred_rank[i] for i in idx]
            labels = sorted(set(yt) | set(yp))
            m = evaluate(yt, yp, labels)
            rows.append({
                "threshold": t,
                "pct_assigned": pct_assigned,
                "accuracy_on_assigned": m["accuracy"],
                "precision_macro_on_assigned": m["precision_macro"],
                "recall_macro_on_assigned": m["recall_macro"],
                "f1_macro_on_assigned": m["f1_macro"],
            })
        else:
            rows.append({
                "threshold": t, "pct_assigned": 0.0,
                "accuracy_on_assigned": None, "precision_macro_on_assigned": None,
                "recall_macro_on_assigned": None, "f1_macro_on_assigned": None,
            })
    return rows


def print_sweep(rows: List[Dict]) -> None:
    print(f"{'threshold':>10} {'%assigned':>10} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for r in rows:
        acc = f"{r['accuracy_on_assigned']:.3f}" if r["accuracy_on_assigned"] is not None else "NA"
        prec = f"{r['precision_macro_on_assigned']:.3f}" if r["precision_macro_on_assigned"] is not None else "NA"
        rec = f"{r['recall_macro_on_assigned']:.3f}" if r["recall_macro_on_assigned"] is not None else "NA"
        f1 = f"{r['f1_macro_on_assigned']:.3f}" if r["f1_macro_on_assigned"] is not None else "NA"
        print(f"{r['threshold']:>10} {r['pct_assigned']*100:>9.1f}% {acc:>10} {prec:>10} {rec:>10} {f1:>10}")

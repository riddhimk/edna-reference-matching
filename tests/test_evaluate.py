from src.evaluate import evaluate, sweep_thresholds


def test_evaluate_perfect_predictions():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    m = evaluate(y_true, y_pred, labels=["A", "B"])
    assert m["accuracy"] == 1.0
    assert m["precision_macro"] == 1.0
    assert m["recall_macro"] == 1.0


def test_evaluate_confusion_matrix_shape():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "A", "A", "B"]
    m = evaluate(y_true, y_pred, labels=["A", "B"])
    assert m["confusion_matrix"].shape == (2, 2)
    assert 0.0 < m["accuracy"] < 1.0


def test_sweep_thresholds_pct_assigned_decreases_with_higher_threshold():
    y_true = ["A", "B", "A", "B", "A"]
    y_pred = ["A", "B", "A", "B", "B"]
    similarities = [0.95, 0.85, 0.75, 0.65, 0.55]
    rows = sweep_thresholds(y_true, y_pred, similarities, thresholds=[0.5, 0.7, 0.9])
    pct_by_threshold = {r["threshold"]: r["pct_assigned"] for r in rows}
    assert pct_by_threshold[0.5] >= pct_by_threshold[0.7] >= pct_by_threshold[0.9]


def test_sweep_thresholds_handles_no_reads_passing():
    rows = sweep_thresholds(["A"], ["A"], [0.1], thresholds=[0.99])
    assert rows[0]["pct_assigned"] == 0.0
    assert rows[0]["accuracy_on_assigned"] is None

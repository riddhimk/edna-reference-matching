from src.open_set_routing import split_for_open_set, get_unclassified_read_ids

SAMPLE_RESULTS = [
    {"Read_ID": "r1", "Confidence_Decision": "assigned"},
    {"Read_ID": "r2", "Confidence_Decision": "unclassified"},
    {"Read_ID": "r3", "Confidence_Decision": "assigned"},
    {"Read_ID": "r4", "Confidence_Decision": "unclassified"},
]


def test_split_for_open_set():
    assigned, unclassified = split_for_open_set(SAMPLE_RESULTS)
    assert [r["Read_ID"] for r in assigned] == ["r1", "r3"]
    assert [r["Read_ID"] for r in unclassified] == ["r2", "r4"]


def test_get_unclassified_read_ids():
    assert get_unclassified_read_ids(SAMPLE_RESULTS) == ["r2", "r4"]

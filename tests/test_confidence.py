import pytest

from src.confidence import decide


def test_decide_raises_without_threshold():
    # threshold must never silently default -- this is intentional, not a bug
    with pytest.raises(ValueError):
        decide(similarity=0.95, margin_to_2nd=0.1, threshold=None)


def test_decide_below_threshold_is_unclassified():
    assert decide(similarity=0.5, margin_to_2nd=0.2, threshold=0.7) == "unclassified"


def test_decide_above_threshold_is_assigned():
    assert decide(similarity=0.9, margin_to_2nd=0.2, threshold=0.7) == "assigned"


def test_decide_ambiguous_margin_is_unclassified():
    # similarity clears the threshold but the runner-up is nearly as close
    assert decide(similarity=0.9, margin_to_2nd=0.01, threshold=0.7, min_margin=0.05) == "unclassified"


def test_decide_clear_margin_is_assigned():
    assert decide(similarity=0.9, margin_to_2nd=0.2, threshold=0.7, min_margin=0.05) == "assigned"

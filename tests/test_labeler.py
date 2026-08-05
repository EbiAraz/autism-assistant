import numpy as np
from facts import CATEGORY_KEYS
from labeler import SemanticLabeler


def test_assign_labels_multi_label():
    dummy = type("Dummy", (), {"cat_keys": CATEGORY_KEYS})()
    scores = np.array([
        [0.1, 0.5, 0.4],
        [0.35, 0.2, 0.1],
    ], dtype=float)

    result = SemanticLabeler.assign_labels(dummy, scores, threshold=0.3, multi_label=True)
    assert result[0]["top_label"] == CATEGORY_KEYS[1]
    assert result[0]["labels"] == [CATEGORY_KEYS[1], CATEGORY_KEYS[2]]
    assert result[1]["top_label"] == CATEGORY_KEYS[0]
    assert result[1]["labels"] == [CATEGORY_KEYS[0]]


def test_assign_labels_none_when_not_confident():
    dummy = type("Dummy", (), {"cat_keys": CATEGORY_KEYS})()
    scores = np.array([[0.1, 0.2, 0.3]], dtype=float)

    result = SemanticLabeler.assign_labels(dummy, scores, threshold=0.5, multi_label=False)
    assert result[0]["labels"] == ["NONE"]

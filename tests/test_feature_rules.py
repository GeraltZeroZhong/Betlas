"""General feature validity and diagnostic-label separation."""
from betlas.features.extract import assert_no_diagnostic_label_leakage
from betlas.features.geometry import entropy

def test_diagnostic_columns_cannot_populate_labels() -> None:
    assert_no_diagnostic_label_leakage({"fold_label_final": "beta_barrel", "betlas_top_fold": "beta_barrel"})
    try:
        assert_no_diagnostic_label_leakage({"fold_label_final": "betlas_top_fold"})
    except ValueError:
        return
    raise AssertionError("expected diagnostic label leakage to be rejected")


def test_entropy_is_bounded_for_degenerate_inputs() -> None:
    assert entropy(["same", "same"]) == 0.0
    assert 0.0 <= entropy(["a", "b", "b"]) <= 1.0



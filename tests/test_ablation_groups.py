from __future__ import annotations

from betlas.ml.ablations import feature_group_for


def test_new_boundary_features_are_grouped() -> None:
    assert feature_group_for("cz_top2_sheet_order_nonlocal_fraction") == "sheet_sequence_topology"
    assert feature_group_for("cz_jelly_roll_order_nonlocal_score") == "sheet_sequence_topology"
    assert feature_group_for("cz_contact8_seq_gap_mean") == "contact_graph"
    assert feature_group_for("cz_sheet_pair_bilayer_score") == "sheet_pair_packing"

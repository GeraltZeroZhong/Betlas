import pandas as pd

from betlas.geometry_relations.cohorts import A, H, T, allocate_roles, category


def test_hierarchy_and_legacy_are_distinct():
    assert category("2.60", "2.60.120") == "jelly_roll"
    assert category("2.70", "2.70.10") == "other_architecture"
    assert category("2.70", "2.70.10", legacy=True) == "other_sandwich"
    assert category("2.30", "2.30.10") != "jelly_roll"


def test_roles_use_full_mother_pdb_and_keep_whole_h():
    rows = [
        {
            "record_id": f"d{i}",
            "pdb_id": f"p{i}",
            A: "2.60",
            T: "2.60.1",
            H: f"2.60.1.{i}",
            "feature_status": "ok",
        }
        for i in range(9)
    ]
    frame = pd.DataFrame(rows)
    panel = frame.iloc[:1]
    roles, _ = allocate_roles(frame, frame, panel)
    held = roles.loc[roles.role.eq("confirmation")].iloc[0]
    bridge = dict(rows[0], record_id="unselected_bridge", pdb_id=held.pdb_id)
    mother = pd.concat([frame, pd.DataFrame([bridge])], ignore_index=True)
    after, summary = allocate_roles(frame, mother, panel)
    assert after.set_index("record_id").loc[held.record_id, "role"] == "confirmation_pdb_purged"
    assert after.loc[after.engineering_h, "role"].eq("discovery").all()
    assert summary["largest_component_domains"] == 2

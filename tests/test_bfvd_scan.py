from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.reproducibility.readouts.bfvd.core import (
    _build_audit_table,
    _metadata_qc_mask,
    _parse_ca_records,
    _split_status,
)


def test_bfvd_split_status_uses_metadata_and_model_suffix():
    assert _split_status("A0A123") == "unspecified"
    assert _split_status("A0A123_2") == "split"
    assert _split_status("A0A123", "1") == "split"
    assert _split_status("A0A123_2", "0") == "unsplit"


def test_bfvd_metadata_qc_respects_confidence_ptm_and_split_flag():
    df = pd.DataFrame(
        [
            {"bfvd_id": "a", "avg_pLDDT": "80", "pTM": "0.7", "splitted": "0"},
            {"bfvd_id": "b", "avg_pLDDT": "60", "pTM": "0.7", "splitted": "0"},
            {"bfvd_id": "c", "avg_pLDDT": "80", "pTM": "0.2", "splitted": "0"},
            {"bfvd_id": "d", "avg_pLDDT": "80", "pTM": "0.7", "splitted": "1"},
        ]
    )

    mask = _metadata_qc_mask(df, min_avg_plddt=70.0, min_ptm=0.45, allow_split=False)
    assert mask.tolist() == [True, False, False, False]

    split_allowed = _metadata_qc_mask(df, min_avg_plddt=70.0, min_ptm=0.45, allow_split=True)
    assert split_allowed.tolist() == [True, False, False, True]


def test_bfvd_pdb_ca_parser_reads_plddt_b_factors(tmp_path: Path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(
        "\n".join(
            [
                "MODEL     1",
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 77.70           C  ",
                "ATOM      2  CA  GLY A   2       2.000   3.000   4.000  1.00 88.80           C  ",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    chain, residues, plddt = _parse_ca_records(pdb)

    assert chain == "A"
    assert [residue.auth_seq_id for residue in residues] == [1, 2]
    assert plddt == [77.7, 88.8]


def test_bfvd_audit_table_contract_uses_diagnostics_without_labels():
    features = pd.DataFrame(
        [
            {
                "record_id": "A0A",
                "bfvd_id": "A0A",
                "UniRef100": "UniRef100_A0A",
                "taxid": "10239",
                "lineage": "Viruses;test",
                "length": 120,
                "split_status": "unsplit",
                "avg_pLDDT": "84.0",
                "pTM": "0.71",
                "BFVD_version": "BASE",
                "betlas_beta_residue_fraction": 0.42,
                "parse_status": "eligible_domain_like_beta_rich",
                "annotation_text": "hypothetical protein",
                "bfvd_quality_flags": "",
                "bfvd_high_confidence_readout_set": 1,
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "record_id": "A0A",
                "betlas_rule_top1_label": "jelly_roll",
                "betlas_rule_top2_label": "beta_sandwich",
                "betlas_rule_probability_margin": 0.04,
                "betlas_rule_probability_entropy": 0.81,
                "betlas_sandwichness": 0.58,
                "betlas_jelly_rollness": 0.72,
                "betlas_barrel_likeness": 0.36,
                "betlas_mixed_topology_flag": 1,
                "betlas_topology_ambiguity_score": 0.66,
            }
        ]
    )

    audit = _build_audit_table(features, diagnostics, foldseek_context=None)

    assert audit.loc[0, "Betlas_top1"] == "jelly_roll"
    assert audit.loc[0, "top2"] == "beta_sandwich"
    assert audit.loc[0, "review_queue_bucket"] == "high_jelly_rollness_high_ambiguity"

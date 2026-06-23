# Betlas Examples

`mini.cif` is a small synthetic mmCIF used for smoke testing the public
single-structure workflow. It contains one chain (`A`) and four annotated beta
strands in `_struct_sheet_range`, which is the mmCIF secondary-structure record
used by Betlas grammar and slicing workflows.
It is intentionally minimal and is not a DSSP readout fixture; for
`beta-barrel-detection` or `beta-barrel-staves`, use a full PDB/mmCIF with the
standard atom-site fields required by Biopython/DSSP.

Run the commands below from the repository root. The `PYTHONPATH=src` prefix is
relative to that root.

```bash
mkdir -p runs/examples
PYTHONPATH=src python -m betlas extract-features \
  --structure examples/mini.cif \
  --chain A \
  --out runs/examples/mini_features.csv
PYTHONPATH=src python -m betlas grammar score \
  --features runs/examples/mini_features.csv \
  --out runs/examples/mini_rule_scores.csv
PYTHONPATH=src python -m betlas slice examples/mini.cif \
  --chain A \
  --out runs/examples/mini_slices.csv \
  --points-out runs/examples/mini_slice_points.csv \
  --summary-out runs/examples/mini_slice_summary.json
```

Expected status:

- `mini_features.csv` contains one row with `record_id=mini_A` and
  `betlas_parse_ok=1`.
- `mini_rule_scores.csv` contains `betlas_top_fold`, `betlas_rule_margin`, and
  one `betlas_rule_score_<fold>` column for each Betlas fold label.
- `mini_slice_points.csv` includes residue-traceable columns such as
  `auth_seq_id`, `residue_uid`, `strand_id`, and `sheet_id`.

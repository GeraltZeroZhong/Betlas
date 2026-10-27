# Betlas examples

`mini.cif` is a synthetic mmCIF for trying the single-structure workflow.
It contains one chain (`A`) and four annotated beta
strands in `_struct_sheet_range`, which is the mmCIF secondary-structure record
used by Betlas grammar and slicing workflows.
For `beta-barrel-detection` or `beta-barrel-staves`, supply a full PDB/mmCIF
with the standard atom-site fields required by Biopython/DSSP.

The geometry is constructed analytically. Four straight strands sit at xy
positions `(10, 0)`, `(0, 10)`, `(-10, 0)`, and `(0, -10)` Angstrom. Each has
four CA points at z coordinates `0`, `2`, `4`, and `6` Angstrom. Residue numbers
are `10 * strand_index + point_index + 1`, with zero-based indices. The chain,
alanine labels, sheet annotation, and unit occupancies are synthetic format
fields generated for this construction. The package contains an identical copy
for `betlas examples`.

Run the commands below from the repository root. The `PYTHONPATH=src` prefix is
relative to that root.

```bash
mkdir -p outputs/examples
PYTHONPATH=src python -m betlas extract-features \
  --structure examples/mini.cif \
  --chain A \
  --out outputs/examples/mini_features.csv
PYTHONPATH=src python -m betlas grammar list
PYTHONPATH=src python -m betlas slice examples/mini.cif \
  --chain A \
  --out outputs/examples/mini_slices.csv \
  --points-out outputs/examples/mini_slice_points.csv \
  --summary-out outputs/examples/mini_slice_summary.json
```

Outputs:

- `mini_features.csv` contains one row with `record_id=mini_A` and
  `betlas_parse_ok=1`.
- The feature table retains the 136 current descriptors, with unavailable measurements left missing.
- `mini_slice_points.csv` includes residue-traceable columns such as
  `auth_seq_id`, `residue_uid`, `strand_id`, and `sheet_id`.

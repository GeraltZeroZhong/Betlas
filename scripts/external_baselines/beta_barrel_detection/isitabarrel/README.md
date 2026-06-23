# IsItABarrel Structure-Map Baseline

This adapter treats
[SluskyLab/isitabarrel](https://github.com/SluskyLab/isitabarrel) as an
external comparison method for Betlas beta-barrel detection evaluation. The
adapter name is `isitabarrel_structure_map` because it runs IsItABarrel
heuristics on contact maps derived from PDB/CIF structures.

The upstream script is not vendored. This directory only contains the
invocation layer, structure-to-contact-map conversion, and normalized output
parser.

## Expected Inputs

- a protein-id list, one id per line or as the first tab-separated column
- a directory containing one contact-map pickle per id, named `<id>.pkl`
- an `isitabarrel.py` checkout, passed with `--script` or
  `ISITABARREL_SCRIPT`

The normalized CSV contains:

- `baseline`: `isitabarrel_structure_map`
- `sample_id`: upstream `MAP_NAME`
- `result`: `BARREL` when the selected score is greater than zero, otherwise
  `NON_BARREL`
- `score`: selected decision score, defaulting to `CC2_TO_H4`

## Structure Map Workflow

```bash
PYTHONPATH=.:src python scripts/external_baselines/beta_barrel_detection/isitabarrel/structure_map.py \
  path/to/structures \
  --out-dir eval_outputs/isitabarrel_structure_map \
  --script /path/to/isitabarrel.py \
  --out eval_outputs/isitabarrel_structure_map.csv
```

The generator writes:

- `maps/<sample_id>.pkl`: NumPy `float32` `L x L` contact maps
- `protid_list.tsv`: sample ids for the upstream script
- `residue_mapping.csv`: matrix index to source residue mapping

The default map uses CA-CA contacts within 8.0 Angstrom, masks contacts with
sequence distance of two residues or less, and skips chains with fewer than 15
CA residues to avoid upstream indexing failures on very short chains.

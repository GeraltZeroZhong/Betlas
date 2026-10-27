# IsItABarrel contact-map adapter

For the specialist comparison:

```bash
python -m scripts.external_baselines.beta_barrel_detection.isitabarrel.specialist \
  structure.cif --sample-id example --chain A --script /path/to/isitabarrel.py \
  --python /path/to/python --out-dir output/isitabarrel-example
```

This selects exactly one model/author chain and records native logs and a
normalized `result.json`. Errors retain an unavailable status. Its normalized
`score` is null: native `CC2_TO_H4` combines closing-contact magnitude and
alternative hairpin counts, which supports the native hard call. The native
value remains in `native_result`; prediction uses the published `> 0` hard call.
AP and ROC ranking require a suitable continuous score.
This variant derives contact maps from coordinates. The original method targets
bacterial transmembrane barrels through evolutionary contact prediction;
membrane insertion requires separate evidence. Contact-map indices follow
observed CA residue order, matching the reused Cooper-Beta comparison
implementation. The mapping preserves gaps and insertion codes, with missing
positions left unfilled.

This adapter runs the official
[IsItABarrel implementation](https://github.com/SluskyLab/isitabarrel), which must
be obtained and installed separately under its upstream license. It also converts
structures into the contact-map format consumed by that implementation.

```bash
python -m scripts.external_baselines.beta_barrel_detection.isitabarrel.structure_map \
  input_structures --script /path/to/isitabarrel.py \
  --out-dir output/isitabarrel --out output/isitabarrel/predictions.csv
```

Contact maps use CA distances in Angstrom, a configurable 8 Angstrom cutoff,
and zeroed contacts at sequence distance two or less. Each map is a float32
matrix stored in the upstream pickle format. The default 15-residue minimum
avoids an upstream indexing failure on short chains. These maps represent
contacts in the supplied coordinates.
`contact_maps` generates the maps, protein-ID list, and residue mapping alone.

For existing contact maps:

```bash
python -m scripts.external_baselines.beta_barrel_detection.isitabarrel.runner \
  protein_ids.txt contact_maps --script /path/to/isitabarrel.py \
  --work-dir output/isitabarrel-work --out output/isitabarrel.csv
```

`--script` or `ISITABARREL_SCRIPT` selects the upstream script; `--python` selects
its Python environment. Additional upstream options follow `--`. Supply contact
maps you trust: their pickle format is consumed by the upstream Python program.
The adapter parses `results.tsv`, requires `MAP_NAME` and the chosen score column,
and assigns `BARREL` to scores greater than zero. The default decision column is
`CC2_TO_H4`; `--decision-column` can select another upstream score.
Normalized CSV rows contain `sample_id`, `result`, `score`, `decision_column`, and
available `CC2`, `BSS`, `H4`, and ratio scores.

```bash
python -m scripts.external_baselines.beta_barrel_detection.isitabarrel.evaluate_dataset \
  --positive-dir positive_structures --negative-dir negative_structures \
  --script /path/to/isitabarrel.py --out-dir output/isitabarrel-evaluation \
  --tag example
```

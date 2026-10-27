# Foldseek structure search

This adapter calls [Foldseek](https://github.com/steineggerlab/foldseek), prepares
chain-level structure queries, and summarizes structure-search hits. Install
Foldseek separately under its upstream license and supply the reference
structures. This directory contains the Betlas integration code.

```bash
python -m scripts.external_baselines.beta_barrel_detection.foldseek.structure_search \
  input_structures --reference-structures reference_structures \
  --foldseek /path/to/foldseek --out-dir output/foldseek \
  --out output/foldseek/predictions.csv
```

Use `runner` for pre-existing queries and a target database:

```bash
python -m scripts.external_baselines.beta_barrel_detection.foldseek.runner \
  query_structures target_database --foldseek /path/to/foldseek \
  --out output/foldseek.csv -- --threads 2
```

`--foldseek` selects the executable; `FOLDSEEK_BIN` or the `foldseek` command on
`PATH` can also supply it. Options after `--` are passed as individual arguments
to Foldseek. `--work-dir` retains raw hits and intermediate files for `runner`.
The `structures` module prepares chains as an independent step.

The default alignment type is 1 (TMalign). The adapter's configurable decision
rule requires query and target coverage of at least 0.6 and
`min(qtmscore, ttmscore) >= 0.5`. These defaults define the adapter's hit policy;
a calibrated beta-barrel classifier requires separate validation. Set
`--score-mode`, `--score-threshold`,
`--min-query-coverage`, and `--min-target-coverage` for the intended application.
A reference collection containing the relevant structures is required.

The hit table uses Foldseek's headerless format:
`query,target,qlen,tlen,alnlen,qcov,tcov,qtmscore,ttmscore,alntmscore,evalue,bits`.
Normalized CSV rows contain `sample_id`, `result`, `score`, the decision rule,
coverage thresholds, hit counts, the best target, and its alignment scores.
Expected query IDs supplied to the API or `--query-id-list` retain no-hit queries
as `NON_BARREL`. Structure workflows supply those IDs automatically.

```bash
python -m scripts.external_baselines.beta_barrel_detection.foldseek.evaluate_dataset \
  --positive-dir positive_structures --negative-dir negative_structures \
  --reference-dir reference_structures --foldseek /path/to/foldseek \
  --out-dir output/foldseek-evaluation --tag example
```

Dataset evaluation excludes reference hits whose filename prefix before the first
underscore matches that of the query, case-insensitively. This supports structure
collections named by source accession; use distinct prefixes for unrelated inputs.
`--reference-dir` defaults to the explicitly supplied positive directory. Reported
metrics therefore depend on that reference choice and exclusion policy.

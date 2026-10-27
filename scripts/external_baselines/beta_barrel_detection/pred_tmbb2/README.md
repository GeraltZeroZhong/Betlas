# PRED-TMBB2 single-sequence adapter

For the specialist comparison:

```bash
python -m scripts.external_baselines.beta_barrel_detection.pred_tmbb2.specialist \
  structure.cif --sample-id example --chain A --juchmme-dir /path/to/juchmme \
  --out-dir output/pred-example
```

Supply the complete declared polymer sequence from mmCIF or PDB SEQRES. The
Python `run_sample` API also accepts an explicit sequence with its provenance.
Ambiguous or noncanonical sequences retain an unavailable status and their
original sequence. This comparator uses sequence input; the geometry methods
consume coordinates. The default membrane-topology task evaluates complete
chains. Domain crops require a separately defined task.

The specialist adapter retains native logs and a normalized `result.json` with
the `LP` topology-derived hard call (three or more membrane-strand runs),
`prediction: 0|1`, and `score: null`. Native topology counts and log-probability
fields remain auxiliary measurements. Calibrating barrel probabilities requires
a separate procedure. Failed runs retain an unavailable status. These settings
reproduce the local single-sequence comparator; the homology-enriched web server
uses a different workflow.

This adapter runs PRED-TMBB2 locally through
[JUCHMME](https://github.com/pbagos/juchmme). The
[PRED-TMBB2 project](https://hannibal.dib.uth.gr/PRED-TMBB2/) provides the method's
context. Obtain the Java implementation and model files separately under their
upstream terms. All execution uses the supplied local installation.

```bash
python -m scripts.external_baselines.beta_barrel_detection.pred_tmbb2.runner \
  input.fasta --juchmme-dir /path/to/juchmme \
  --out output/pred-tmbb2.csv
```

`--juchmme-dir` or `PRED_TMBB2_JUCHMME_DIR` selects a local release or checkout.
It must contain compiled classes in `bin`, transition and emission tables at
`tables/A_TMBB2_TRAINED` and `tables/E_TMBB2_TRAINED`, the model at
`models/tmbb2.mdel`, and configuration at `conf/conf.tmbb`. The Python
`run_baseline` API also accepts explicit `classpath`, `transitions`, `emissions`,
`model`, and `config` paths for another installation layout. `--java` selects Java.

The adapter parses JUCHMME's `ID`, `CC`, and topology records. `--prediction-field`
selects `LP` (default) or `VP`; `--min-tm-strands` sets the number of consecutive
`M` regions required for the adapter's `BARREL` decision, defaulting to three.
That threshold is an adapter policy and is recorded in every result. The local
single-sequence workflow differs from a homology-enriched web-server workflow.

Normalized CSV rows contain `sample_id`, `result`, `score`, `tm_strands`,
`decision_rule`, `prediction_field`, `topology`, and available reliability,
algorithm-score, length, and log-probability fields.

To extract sequences from structure chains and run the same method:

```bash
python -m scripts.external_baselines.beta_barrel_detection.pred_tmbb2.structure_sequence \
  input_structures --juchmme-dir /path/to/juchmme \
  --out-dir output/pred-tmbb2 --out output/pred-tmbb2/predictions.csv
```

The `sequences` command generates FASTA and residue mappings as a standalone
preparation step. Only observed protein residues with CA atoms enter the FASTA;
supply complete protein sequences directly to `runner`.

```bash
python -m scripts.external_baselines.beta_barrel_detection.pred_tmbb2.evaluate_dataset \
  --positive-dir positive_structures --negative-dir negative_structures \
  --juchmme-dir /path/to/juchmme --out-dir output/pred-tmbb2-evaluation \
  --tag example
```

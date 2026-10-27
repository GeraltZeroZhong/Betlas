# Cooper-Beta native hard-call adapter

This adapter invokes the separately installed [Cooper-Beta](https://github.com/GeraltZeroZhong/Cooper-Beta)
CLI. It preserves the native default strand-graph conjunction and returns its
hard decision. The learned adapter below uses a separately supplied fitted model.

```bash
python -m scripts.external_baselines.beta_barrel_detection.cooper_beta.runner \
  structure.cif --sample-id example --chain A --model-index 0 \
  --python /path/to/environment/bin/python --source-root /path/to/Cooper-Beta \
  --dssp /path/to/mkdssp --out-dir output/cooper-example
```

The output directory must be new. One selected coordinate model and author chain
are supplied to the native program. Author residue numbering, coordinates and,
for mmCIF, declared polymer categories are preserved. The source file remains
unchanged. `native.csv`, the native manifest, stdout/stderr, and `result.json` are
retained. The latter records `status`, `prediction`, `score`, input selection,
command, source commit (when a checkout is supplied), and elapsed wall time.

Successful native `BARREL`/`NON_BARREL` calls become predictions 1/0 with
`score: null`. Failed or degraded runs retain an unavailable status. The endpoint
is chain-level beta-barrel-like geometry. Membrane insertion and localization
require separate evidence.

## Frozen learned model

The separately named `cooper_beta_learned` adapter reuses a successful native
feature result and runs inference with the exact trusted local fitted model and
its tuning metadata:

```bash
python -m scripts.external_baselines.beta_barrel_detection.cooper_beta.learned \
  output/cooper-example/result.json --model /path/to/tuned_logistic.joblib \
  --metadata /path/to/tuning_summary.json --selection deployment \
  --python /path/to/matching-sklearn/bin/python --out-dir output/cooper-learned
```

Use a compatible scikit-learn environment. The worker fails on
serialization-version warnings. The native four-feature order and exact
threshold are read from the metadata.
`--selection outer_N` selects an explicitly supplied outer-fold artifact and its
recorded threshold. The normalized score is its positive-class probability.
The operator must supply the model corresponding to the declared selection.

The deployment model was fitted on its recorded cohort. Evaluate held-out
performance on independent inputs or with the appropriate original outer-fold
model, and check relatedness and overlap. Model-loading/scoring time is reported
separately from native graph extraction. Account for the shared extraction cost
once.

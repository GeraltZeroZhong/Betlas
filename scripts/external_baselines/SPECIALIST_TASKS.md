# Specialist structure recognition tasks

The adapters support local execution on caller-supplied structures and sequences.
Each result preserves the method variant, input regime, native output, execution
status, and the meaning of any score. Supply third-party software and model files
separately. The commands run locally with those supplied assets.

## Endpoints

| Task | Unit and reference definition | Applicable methods |
| --- | --- | --- |
| `beta_solenoid` | Exact structure-domain coordinates; declared solenoid architecture reference | Betlas learned solenoid classifier, SOLeNNoID domain aggregation, RAPHAEL native result import, Foldseek labelled-reference readout |
| `barrel_geometry` | Selected chain and coordinate model; independently curated barrel-like geometry | Betlas barrel readout, Cooper-Beta direct/frozen learned, IsItABarrel structure-map adaptation, Foldseek labelled-reference readout |
| `membrane_beta_barrel` | Selected chain with separately supported membrane-barrel labels | PRED-TMBB2 single-sequence JUCHMME; other methods may be assessed as explicitly named transfers |

Barrel geometry and membrane localization require their own reference labels.
A soluble geometric barrel is negative for the membrane-barrel endpoint. Supply
membrane evidence and residue-level solenoid annotations separately from CATH
architecture labels. Keep uncertain labels separate from the definite binary
endpoint.

IsItABarrel's original workflow targets bacterial transmembrane barrels using
evolutionary contact predictions. The coordinate-contact-map adapter is an
explicit adaptation. PRED-TMBB2's local JUCHMME variant uses a single sequence and
an LP topology rule. The full web-server workflow also offers signal-peptide,
Pfam and OMPdb processing.

## Native methods

See the [solenoid adapters](solenoid/README.md) for official-model SOLeNNoID CPU
execution and saved RAPHAEL output parsing. RAPHAEL import reads existing
predictions; new native predictions require a verified local executable.

See the [Cooper-Beta adapter](beta_barrel_detection/cooper_beta/README.md),
[IsItABarrel adapter](beta_barrel_detection/isitabarrel/README.md), and
[PRED-TMBB2 adapter](beta_barrel_detection/pred_tmbb2/README.md) for native local
commands. Select the same chain/model coordinates for coordinate methods. For
PRED-TMBB2 supply the complete declared polymer sequence, retaining unresolved
residues; a sequence derived only from observed coordinates changes the input.
The optional Cooper-Beta learned adapter scores native graph features with a
caller-supplied frozen model and its original metadata in a matching sklearn
environment. It uses the recorded probability threshold for inference. Use
held-out examples to evaluate the deployment model independently of its fitting
cohort.

Betlas provides a native barrel readout and checkpoint-based solenoid prediction:

```bash
betlas readout beta-barrel-detection inputs/chain.cif --chain A \
  --workers 1 --prep 1 --out outputs/betlas-barrel.csv \
  runtime.dssp_bin_path=/path/to/mkdssp
betlas predict solenoid \
  --features inputs/geometry.csv --out outputs/betlas-solenoid.csv
```

The solenoid feature table must contain the current 136 descriptors measured
from the same domain crop and DSSP annotation context. The model applies the
preprocessing and logistic-regression parameters stored in its checkpoint;
see the [model guide](../../docs/models.md) for its training cohort and measurement
requirements. For the specialist evaluator, align predictions with the declared
`sample_id` values and map `probability` to `score`, preserving `prediction` and
`status`. The barrel readout emits its native decision and an uncalibrated
heuristic score; preserve gate rejections as native decisions when execution
succeeded. Freeze model and readout choices before test evaluation.

Import a Betlas barrel CSV through an explicit sample map with
`sample_id,source_path,chain_id` (the path identifies the exact file analyzed):

```bash
python -m betlas.benchmarks.specialists betlas-barrel-scores \
  --native-csv outputs/betlas-barrel.csv --samples inputs/barrel-samples.csv \
  --out outputs/betlas-barrel-scores.csv
```

Missing source/chain results remain unavailable. A native gate rejection after
successful analysis remains a negative decision.

## Labelled-reference Foldseek

The same local binary retrieval interface serves both structure tasks. Query
CSV columns are `sample_id,structure_path,chain_id,pdb_id,group_id`, optionally
`model_id` (zero-based, default 0). Reference CSV additionally requires binary
`target`. Paths may be absolute or relative to the CSV. Query labels are unused.
For a non-PDB source, use its stable source accession in `pdb_id` and document
that identifier convention. Assign `group_id` from the declared biological
relatedness groups so related samples share a group.

Both reference classes are required. The adapter rejects overlap by sample,
source accession, biological group, and exact coordinate source/model/chain.
Full-catalog links and upstream training membership remain caller responsibilities.
Coordinates are exported for the selected model and chain, with the exported
chain named `A`; original identities are retained in the input inspection.

```bash
python -m scripts.external_baselines.structure_retrieval \
  --queries inputs/queries.csv --references inputs/reference.csv \
  --task beta_solenoid --foldseek /path/to/foldseek \
  --output-dir outputs/solenoid-foldseek --threads 1 --timeout-seconds 300
```

Use `--task barrel_geometry` for the geometry task. `--exhaustive` bypasses the
prefilter and is suitable for a tiny smoke collection; its cost grows with all
query/reference pairs. With prefiltering enabled, results describe retrieved
hits. The candidate cap equals the available reference count; the prefilter
still determines which pairs reach alignment. Intermediate databases use
Linux temporary storage by default; `--scratch-dir` must support symbolic links.
Persistent exports, raw hit tables, logs and normalized results remain in the
specified output directory.

The adapter defaults to global TM-align (`--alignment-type 1`) on CPU.
Use `--alignment-type 2` for Foldseek's native default local 3Di+AA alignment,
with TM scores computed for the returned alignments. This changes the alignment
algorithm and may change coverage and predictions; freeze the choice before
evaluation. Type 1 reports average TM in its `evalue` field, whereas type 2 uses
true E-values. The adapter uses the explicit TM fields for either mode.
Eligible hits require both query and target coverage of at least 0.6 by default.
Define `s = min(qtmscore, ttmscore)`. The readout is:

```text
score = maximum positive-reference s - maximum negative-reference s
prediction = 1 if score > 0 else 0
```

Equal scores predict negative. An absent class among eligible returned hits has
zero evidence, with its absence recorded explicitly. If neither class has an
eligible hit, the result is unavailable. A missing reference class after
preparation also prevents
classification. The difference of similarities is a derived classification
score. Probability calibration requires separate validation. Fix reference size
and composition before evaluation because both affect the score. This protocol
uses both reference classes; the positive-library-only Foldseek barrel detector
uses a separate decision rule.

## Shared evaluation and coverage

The [specialist evaluator](../../src/betlas/benchmarks/SPECIALISTS.md) imports
per-sample normalized JSON through a CSV with `sample_id,result_path`. For
Foldseek, this is the generated `results.csv`.

```bash
python -m betlas.benchmarks.specialists import-scores \
  --results outputs/foldseek/results.csv --method Foldseek \
  --out outputs/foldseek-scores.csv
python -m betlas.benchmarks.specialists evaluate \
  --labels inputs/test-labels.csv --scores outputs/foldseek-scores.csv \
  --method Foldseek --native-decisions --out outputs/foldseek-evaluation.json
```

Cooper-Beta direct, IsItABarrel's hybrid closing/hairpin rule and the local
PRED-TMBB2 topology rule provide hard decisions. Their normalized `score` is null
and `prediction` is 0 or 1. Use `--native-decisions`; AP and ROC AUC stay null
without continuous scores. Auxiliary native magnitudes remain in raw outputs.
For continuous scores, fixed or development-only calibrated thresholds remain
available. Report native and calibrated operating points separately.

All planned queries remain in the evaluation denominator, including unavailable
results. Count a correct negative only when a method returns a valid negative
decision for a negative reference label. Ranking metrics use available continuous
scores and report their denominator. Small synthetic or biological smoke runs
check execution and mapping. Accuracy and independence require evaluation on
a suitable cohort with relatedness and training-overlap checks.

## Verification

```bash
python -m pytest tests/test_specialist_evaluation.py \
  tests/test_specialist_structure_retrieval.py \
  tests/test_external_barrel_specialists.py tests/test_external_solenoid.py
```

Public tests use synthetic inputs. Keep real structures, trained models, output
tables and research reports outside the public source tree.

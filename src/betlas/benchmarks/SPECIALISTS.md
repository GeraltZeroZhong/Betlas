# Comparing specialist beta-structure methods

Choose a shared endpoint before comparing methods. A CATH topology identifier,
a strand-pairing graph, a residue contact and a solenoid region are different
outputs. The adapters preserve each tool's native endpoint and mark unavailable
outputs explicitly.

The [two-task adapter guide](../../../scripts/external_baselines/SPECIALIST_TASKS.md)
also covers barrel geometry, membrane-specific input distinctions, Cooper-Beta,
IsItABarrel, PRED-TMBB2 and labelled-reference Foldseek for either structure task.

Use an environment with Betlas installed. For an uninstalled source checkout,
run the commands below from the repository root with `PYTHONPATH=src`.

## Structure-input solenoid comparison

Betlas, RAPHAEL and SOLeNNoID can be compared on the same domain coordinates
for domain-level solenoid detection. A CATH-derived task can define architectures
2.150 and 2.160 as positive and other Class 2 architectures as reference-negative.
This endpoint measures recognition of those CATH architectures. Reference-negative
structures may still contain local repeat regions.

Use exact domain crops, the same coordinate model and residue identity mapping.
Retain all selected inputs in the cohort, including Betlas extraction failures.
A chain score and a domain score describe different inputs; document domain
cropping when adapting a chain detector.
SOLeNNoID's native author-number gaps affect its padded distance matrix. Its
adapter preserves that input representation and reports unsupported identities.

Method scores have distinct meanings:

* Betlas: the solenoid probability from `betlas predict solenoid`. The model
  applies median imputation and standard scaling to the 136 geometric descriptors
  before logistic regression. Use the same domain coordinates as the other
  methods and document the DSSP input context. The
  [model guide](../../../docs/models.md) describes the checkpoint's training cohort
  and measurement requirements.
* RAPHAEL: the native SVM score. Its total structural score is a separate output.
  Preserve the published strict `score > 0` decision and optional `> 1` high-confidence
  criterion when evaluating those native rules. Interpret these values on the
  SVM score scale.
* SOLeNNoID: the fraction of observed, scored residues called beta-solenoid.
  A 0.5 fraction threshold adapts the paper's chain aggregation to domains.
  Keep the denominator and residue coverage. Native non/beta/alpha-beta/alpha
  outputs remain separate. Residue-level evaluation requires its own reference
  labels.

Report native decisions and, separately, development-calibrated decisions when
appropriate. Calibration must precede test access and use the same objective for
each method. The module below checks sample, homology-group and PDB separation
between development and test. PDB identities must come from the full domain
catalog when purging cross-split structures so that links through unselected
domains are included. Check upstream training membership and mark unknown overlap
when assessing whether test structures were unseen by pretrained models.

### Domain-score interface

Labels CSV has `sample_id,target,split,group_id,pdb_id`, with binary `target`,
`development` or `test` split, and a biological group such as CATH H. Scores CSV
has `sample_id,score,status`. `ok` requires a finite score; other statuses require
an empty score. Missing rows remain in the cohort with unavailable status. Supply
only the intended split's rows to each command.

For a native hard classifier, scores CSV additionally has `prediction` (0 or 1)
and leaves `score` empty. Evaluate it with `--native-decisions`. These calls count
toward coverage and confusion metrics, while AP and ROC AUC remain unavailable.
Threshold calibration requires continuous scores. A result containing both a
continuous score and a native prediction supports either explicit evaluation mode.

```bash
python -m betlas.benchmarks.specialists import-scores \
  --results inputs/solennoid-results.csv --method SOLeNNoID \
  --out outputs/solennoid-scores.csv
python -m betlas.benchmarks.specialists calibrate \
  --labels inputs/development-labels.csv --scores inputs/development-scores.csv \
  --method example-model --out outputs/threshold.json
python -m betlas.benchmarks.specialists evaluate \
  --labels inputs/test-labels.csv --scores inputs/test-scores.csv \
  --method example-model --calibration outputs/threshold.json \
  --out outputs/evaluation.json
python -m betlas.benchmarks.specialists evaluate \
  --labels inputs/test-labels.csv --scores inputs/raphael-test-scores.csv \
  --method raphael-native --fixed-threshold 0 --operator gt \
  --out outputs/raphael-native.json
```

The results manifest for `import-scores` has `sample_id,result_path`, pointing to
each adapter's normalized JSON, with paths relative to the manifest or absolute.
The importer extracts SOLeNNoID `beta_fraction` or RAPHAEL `svm_score`, checks
sample/method identity, and preserves unavailable states.
It also accepts normalized `Foldseek`, `cooper_beta`, `cooper_beta_learned`,
`isitabarrel_structure_map`, `pred_tmbb2_single_juchmme`
and `betlas_barrel_geometry` results. Use separate evaluation tables for
different task definitions or information regimes.

Evaluation retains an explicit unavailable column in the confusion table.
Balanced accuracy and recalls use all reference records. Precision counts actual
emitted calls; F1 includes unavailable true-class records in recall. AP and ROC
AUC use the score-available subset, report its denominator, and are unavailable
when that subset lacks either class. They summarize score ranking. Group-mean
accuracy is descriptive; uncertainty estimates must account for correlation
among domains.

See the [solenoid adapters](../../../scripts/external_baselines/solenoid/README.md)
for verified upstream interfaces and their execution limits.

## Sequence-input strand pairing and contact comparison

BetaPro, BCov and bbcontacts take sequence-derived information, with different
profile, secondary-structure and coevolution requirements. Their native endpoints
can be compared within an explicitly declared information regime:

* `sequence_predicted_ss`: each documented sequence/profile/MSA pipeline and its
  predicted secondary structure; preprocessing differences are part of the system.
* `sequence_observed_ss`: a separately labelled control using observed secondary
  structure, when supported by the tool.
* `structure_reference`: Betlas or another coordinate-derived annotation. This
  is a structural reference with access to coordinates.

Assess structural agreement against an independent endpoint: the same DSSP
edges used as input would give a circular comparison. Report this assessment
separately from prediction using only sequence information. Label any substituted
MSA/database release as an input variant. Initially restrict sequence-contact
inputs to a single contiguous domain with an explicit sequence-to-coordinate map; concatenating
discontinuous CATH segments creates artificial sequence neighbors.

Native residue contacts and strand partners must remain separate. For a shared
strand endpoint, map each residue call to independently fixed reference-strand
intervals and declare the aggregation before seeing results. Preserve parallel,
antiparallel and unknown orientation, and keep isolated bridges distinct from
sheet-ladder contacts. Dense probabilities, sparse selected paths, hard contacts
and pseudo-energies require evaluation suited to their score semantics.

### Pair-call interface

The `pairs` evaluator consumes:

* A complete, caller-declared eligible reference universe CSV with
  `sample_id,residue_i,residue_j,target,orientation`.
* Emitted calls CSV with `sample_id,residue_i,residue_j,orientation`.
* Execution statuses CSV with `sample_id,status`.

Indices are positive and one-based; for `--unit strand_pair` they denote reference
strand IDs. Pairs are unordered; reverse duplicates and calls outside the
supplied universe produce errors.
Orientations are `parallel`, `antiparallel`, `bridge`, `unknown` or empty. Native
isolated bridges remain a separate type; the supplied reference must define
whether they are positive for the chosen contact endpoint. They have no
parallel/antiparallel orientation and are excluded from orientation truth support.
A successful empty call set needs an explicit `ok` or native `empty` status.
Unsupported inputs, failures and missing MSAs retain their unavailable status.

```bash
python -m betlas.benchmarks.specialists import-pairs \
  --results inputs/bcov-results.csv --unit residue_contact \
  --out outputs/bcov-calls.csv --statuses-out outputs/bcov-status.csv
python -m betlas.benchmarks.specialists pairs \
  --reference inputs/reference-pair-universe.csv --calls inputs/contact-calls.csv \
  --statuses inputs/execution-status.csv --unit residue_contact \
  --input-regime sequence_predicted_ss --out outputs/contact-evaluation.json
```

`import-pairs` uses the same `sample_id,result_path` manifest, one method at a
time. It converts the native JSON hard calls, including bridge labels and valid
empty outputs. Convert a BetaPro dense probability map to hard calls using a
separately declared decision rule before import. Native strand IDs require an
explicit map to shared reference strands before cross-method strand evaluation.

Domain-macro precision/recall/F1 avoid domination by long structures; unavailable
positive domains have zero recall. Undefined denominators remain null. Orientation
recall requires both the contact and its orientation. AP is unavailable for
hard calls.
The [native adapters](../../../scripts/external_baselines/beta_topology/README.md)
preserve each method's original output semantics. Supply independent biological
reference labels and complete MSAs separately.

## Verification

```bash
python -m pytest tests/test_specialist_evaluation.py \
  tests/test_external_beta_topology.py tests/test_external_solenoid.py
```

These tests use synthetic inputs to check identity mapping, threshold leakage,
failure denominators, pair orientation and parser/command contracts. Biological
accuracy requires evaluation on reference-labelled structures. Large inputs,
weights, raw outputs and private research reports belong outside the public
source tree.

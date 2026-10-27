# Structure-input solenoid baselines

These adapters compare solenoid detection from supplied coordinates. They
support residue predictions, domain aggregation and global repeat scores. CATH architecture classification
requires a separate classifier, and residue evaluation requires residue-level
annotations. Use the same single-chain coordinate crop for Betlas and the external
method, with explicit sample identifiers, development/test roles, and target labels.

| Method | Native input/output | Supported integration |
| --- | --- | --- |
| [SOLeNNoID](https://github.com/gnik2018/SOLeNNoID) | Coordinates; per-residue solenoid class scores | Local CPU API worker, native CSV parser, explicit residue mapping and domain aggregation |
| [RAPHAEL](https://protein.bio.unipd.it/raphael/) | Coordinates; global SVM margin and repeat period | Parser for saved native HTML results; local executable not verified |

Supply the upstream software, weights, structures and benchmark labels
separately. Commands use local assets in an existing environment; installation
and server submission remain separate steps.

## SOLeNNoID local execution

The inspected upstream version is commit
`87c13fefdd6acd7cb43831979d89fd11e437716f` (GPL-3.0). Supply its
`solenoid_detector_onehot_v3.py`, the official model H5, and a Python interpreter
with TensorFlow, NumPy, and Biopython. The repository's `requirements.txt` is a
historical macOS Conda export. Prepare a compatible environment for your platform
using the dependencies above.

Run from the Betlas repository root:

```bash
python -m scripts.external_baselines.solenoid.solennoid \
  --structure /path/to/exact-domain.pdb \
  --sample-id example-domain \
  --chain-id A \
  --source-dir /path/to/SOLeNNoID \
  --model-path /path/to/official-checkpoint.h5 \
  --python-executable /path/to/tensorflow-environment/bin/python \
  --output-dir /path/to/new-output-directory \
  --max-span 2048 --timeout-seconds 120
```

The worker imports upstream `Protein` and `Prediction` without editing them,
loads the supplied weights with `compile=False`, and runs on CPU with one
TensorFlow thread for each thread pool. It supplies the removed `np.float`
alias as the identical Python `float` builtin inside the isolated worker.
For Keras 3 only, `--keras3-groups-one-compat` explicitly enables a deserialization
adapter that drops the obsolete `Conv2DTranspose(groups=1)` keyword. Other group
values fail; weights, activations, padding, and convolution parameters remain
unchanged. The compatibility mode is recorded in `worker.json`.

Outputs include the input inspection and identity mapping, raw predictions for
all integer positions, observed-residue predictions, logs, the exact invocation,
runtime metadata, and `result.json`. An existing output directory is rejected to
preserve previous results. A timeout, unsupported input, missing dependency,
or native failure retains an `unavailable` status.
The normalized result records `execution_mode=local_cpu_api_worker`; a successful
execution must additionally have `status=ok`.

The first coordinate model is used. The selected chain must contain protein
residues with CA atoms and at least four atoms per residue, matching upstream's
actual atom filter. Author integer numbering is preserved. Missing integer
positions retain native zero padding, and are excluded from the observed-residue
aggregation. Insertion codes, repeated or nonmonotonic author numbers,
ambiguous residue identities, zero-distance normalization, or residues that
upstream would silently filter are reported as unsupported. The adapter
preserves the supplied residue numbering and atoms.
The explicit `--max-span` resource limit uses last minus first author residue
number plus one, which may greatly exceed the number of observed residues.

Upstream's CLI (`python solennoid.py -i input.pdb -csv`) emits
`chain,index,residue,one_hot,class`. Its export code assumes contiguous residue
numbering and can silently skip failing chains. `parse_native_csv` therefore
requires an explicit complete mapping and rejects gapped numbering, duplicate
identities, missing rows, residue-name disagreement, or label/score disagreement.
The API worker handles gaps by preserving the native matrix and mapping scores
back to observed author numbers. The upstream CLI assigns residues by position.

Normalized rows contain `sample_id,chain_id,residue_number,insertion_code`,
`predicted_class` (0 non-solenoid, 1 beta, 2 alpha/beta, 3 alpha), four native
score channels, and omitted probability mass. The upstream API returns only the
first four of six softmax channels and takes their argmax. These four values
can sum to less than one. The adapter preserves their original values, records
the omitted mass, and follows the native four-channel decision even when the
omitted mass dominates.

The domain-level score is the fraction of observed predicted residues assigned
to beta-solenoid (`beta_fraction`), with `any_solenoid_fraction` also retained.
These fractions aggregate residue-level hard calls. Domain-class probabilities
require separate calibration. Use a stated fixed threshold or select one on
development data only. Evaluate residues against residue-level annotations and
include only observed coordinates in the denominator. Check the model's published
training structures for benchmark overlap when evaluating independent
generalization.

Observed protein residues include canonical selenocysteine (`SEC`) and
pyrrolysine (`PYL`) explicitly, because Biopython's amino-acid lookup omits them.
Their original names, author numbers, coordinates and native model treatment
are preserved.

## RAPHAEL saved-result import

The official result example distinguishes `SVM SCORE` (the classifier margin)
from `Total score` (a geometric statistic). The parser reads the former and
requires a completed result. Positive margins indicate repeats and negative
margins indicate globular structures. Both margins and total scores are
uncalibrated measurements. RAPHAEL detects solenoids in general; beta-solenoid
classification requires a separate readout.

```bash
python -m scripts.external_baselines.solenoid.raphael \
  --native-html /path/to/saved-raphael-result.html \
  --sample-id example-domain --output /path/to/new-result.json
```

Associate each result with the exact submitted coordinate crop and preserve
that input and job provenance, including the chain, domain boundaries and
coordinate version. A new comparison requires matched inputs; historical CATH
3.4 scores retain their original benchmark context. Imported results record
`execution_mode=saved_native_result_import` and
`prediction_execution_verified=false`. Parsing verifies the saved format and
score. Execution on the current input requires separate provenance.

The verified resources are the official website, saved example and advertised
coordinate datasets. The inspected primary sources provided no executable,
source distribution or local command line. The parser supports saved-result
import. Automated local prediction requires a verified local implementation,
and server submission is handled separately.

## Resource planning

`inspect_structure` and `matrix_cost` expose observed residue count, author-number
span, distance-matrix storage, padded storage, tile count, and native reconstructed
prediction-matrix storage. For span `L`, the float64 distance matrix uses `8 L²`
bytes, and upstream reconstructs a six-channel float64 matrix at the next
multiple of 128, requiring approximately `48 L²` bytes. TensorFlow tensors,
model state, temporary arrays, and process overhead add to these numbers.
The official H5 is 23,387,520 bytes. Upstream computes pairwise distances in
Python loops; both observed residue count and numbering span affect runtime.

Measure a small length-stratified local pilot before scheduling a cohort.
One-process-per-structure execution includes TensorFlow/model loading each time.
Measure that cost locally when comparing with published timings from persistent
models. The wrapper runs each structure on CPU; batching and parallel scheduling
require caller coordination.

`worker.NativePredictor` also supports loading the official model once per CPU
process and calling `predict(inspection, output_dir)` repeatedly. It preserves
the same upstream per-structure algorithm and native span output. The caller
must provide a ready inspection, an existing output directory, explicit resource
limits, per-case failure handling, and durable output storage. Validate numerical
and residue-identity agreement with isolated execution before using this API for
a cohort. Reusing the model preserves the per-structure method.

Synthetic parser, mapping, unsupported-input, local-invocation, and compatibility
checks are in `tests/test_external_solenoid.py`.

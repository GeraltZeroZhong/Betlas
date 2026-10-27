# Betlas

Betlas is a Python toolkit for describing beta-rich protein structures through geometry and strand connectivity. It converts C-alpha coordinates and secondary-structure annotations into 136 named measurements of sheet organisation, strand pairing, contact topology, closure and overall shape. Feature tables and residue-linked slices make these properties available for inspection, structural comparison and model training.

Three trained applications use these measurements to recognise solenoids and geometric beta barrels, and to count the beta strands forming a barrel wall.

The name comes from Beta + Atlas. It reflects the idea that beta-structure patterns can become a map: a set of landmarks that can be inspected, joined, and reproduced.

[Quickstart](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/README.md#quickstart) · [Descriptors](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/README.md#what-betlas-measures) · [Applications](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/README.md#structural-applications) · [Python API](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/README.md#python-api)

## Installation

Python 3.10 or newer is required. Install or update the released package from PyPI:

```bash
python -m pip install --upgrade betlas
```

From a source checkout:

```bash
python -m pip install .
```

Structure-based predictions also require a local `mkdssp` executable for secondary-structure annotation. Use the [Conda environment](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/environment.yml), or install DSSP separately and pass `--dssp-binary /path/to/mkdssp`. DSSP 4.6.1 is recommended for inference. Prediction outputs record the annotation version; the [model documentation](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/docs/models.md) describes the training annotations.

The `geometry` extra provides Gemmi, NetworkX and Matplotlib for geometry relationship analysis. The `ml` extra adds XGBoost. Install `.[dev,geometry]` for the complete test suite.

## Quickstart

The bundled example runs offline. It contains four analytically constructed beta strands with four C-alpha points each, arranged in one annotated sheet. Copy it, inspect the chain, extract descriptors and trace the geometry through axial slices:

```bash
betlas examples copy mini --out-dir outputs/example
betlas chains outputs/example/mini.cif

betlas extract-features \
  --structure outputs/example/mini.cif --chain A \
  --out outputs/features.csv

betlas slice outputs/example/mini.cif --chain A \
  --out outputs/slices.csv \
  --points-out outputs/slice_points.csv \
  --summary-out outputs/slice_summary.json
```

`features.csv` contains one record with the 136 descriptor columns plus identifiers and extraction status. `slices.csv` describes informative axial bins; `slice_points.csv` links their coordinates to residues, strands and sheets. The JSON summary records the slicing setup and aggregate geometry.

```python
import pandas as pd
from betlas.features.current import FEATURE_COLUMNS

features = pd.read_csv("outputs/features.csv")
print(features[["record_id", "chain_id", "betlas_parse_ok"]])
print(features.loc[:, list(FEATURE_COLUMNS)].shape)  # (1, 136)

points = pd.read_csv("outputs/slice_points.csv")
print(points[["auth_seq_id", "residue_uid", "strand_id", "sheet_id"]].head())
```

The [example documentation](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/examples/README.md) explains the construction and outputs. This C-alpha fixture illustrates descriptor extraction and slicing. Use a full-atom protein structure for the DSSP-based applications below.

## What Betlas measures

The 136 descriptors form a fixed, ordered feature set organised into 12 families:

| Family | Structural properties |
| --- | --- |
| `composition` | Residue and secondary-structure counts, lengths and fractions |
| `global_shape` | Spatial extent and anisotropy along principal axes |
| `sheet_global` | Sheet sizes, planarity, angular spans and strand-axis dispersion |
| `beta_run_topology` | Lengths, sheet membership and angular arrangement of sequence-contiguous beta runs |
| `sheet_pair_packing` | Separation, plane alignment and contacts between the two largest sheets |
| `sheet_sequence_topology` | Sheet interleaving, strand order and nonlocal pairing |
| `contact_graph` | Contact density, connected components, cycles and sequence locality |
| `axis_closure` | Angular coverage, gaps, radial variation and continuity along an axis |
| `axis_periodicity` | Repeated angular organisation through circular moments of orders 3 to 8 |
| `strand_order` | Strand directions and sequence order relative to axes and neighbours |
| `alpha_shell` | Helix context, radial organisation and alpha/beta alternation |
| `angular_lobes` | Broad angular organisation through circular moments of orders 1 and 2 |

Inspect family descriptions and individual field definitions:

```bash
betlas grammar list
betlas grammar describe sheet_pair_packing
```

```python
from betlas import describe_feature, list_feature_specs

print(len(list_feature_specs()))  # 136
print(describe_feature("betlas_axis_best_score").formula)
```

The field registry supplies definitions, units and missing-value conventions. Extraction retains every descriptor column, including measurements that are unavailable for a particular geometry. For example, a relationship between two sheets requires two suitable sheets. Application checkpoints apply their fitted preprocessing to these missing values. Coordinates are interpreted in angstroms.

Slicing connects aggregate measurements to local structure. Each axial bin summarises the surrounding beta geometry, while its point table retains author residue numbers and strand/sheet membership for inspection in a molecular viewer. The lower-level `readout` commands expose additional geometric measurements and heuristics.

## Inputs and outputs

| Workflow | Input | Output |
| --- | --- | --- |
| `extract-features` | An annotated mmCIF chain or a labelled structure collection | The 136 descriptors, identifiers and extraction status |
| `slice` | An annotated mmCIF chain | Axial bins, optional residue-linked points and a summary |
| `predict` | A PDB/mmCIF chain or a task-specific feature table | Predictions, probabilities and model/applicability information |
| CATH benchmarks | Domain annotations, features and evaluation groups | Coverage, metrics and held-out predictions |

For raw single-structure extraction and slicing, supply `.cif`, `.mmcif` or gzip-compressed equivalents with usable beta-sheet annotations. Missing sheet records produce an annotation error. The application and frozen CATH workflows assign secondary structure with DSSP, including annotation of structures that contain no beta segments.

Use the author chain identifier and a zero-based `--model-id` (default `0`). Numeric author residue ranges select domains; prepare insertion-code selections upstream. Label-table extraction reads local structures and downloads missing mmCIF files from RCSB. See the [benchmark workflow](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/src/betlas/benchmarks/README.md) for domain selection, annotations and batch extraction.

## Structural applications

Supply a local full-atom PDB or mmCIF structure and its author chain identifier:

```bash
betlas predict solenoid --structure protein.cif --chain A --out solenoid.csv
betlas predict barrel --structure protein.cif --chain A --out barrel.csv
betlas predict barrel-strands --structure protein.cif --chain A --out strand_count.csv
```

Solenoid recognition can select a domain with `--residue-ranges 10-180:A`. Barrel recognition and strand counting operate on the selected chain. Gzip-compressed structures are supported.

Strand counting combines the 136 descriptors with 16 layer/radial measurements. Its default workflow first checks the barrel-recognition model, then predicts the number of strands forming the barrel wall. The trained classes are 8, 10, 12, 14, 16, 18 and 22 strands. For an independently established barrel, add `--assume-barrel` to count directly.

Each command writes predictions with model and applicability information. Choose a `.json` output filename for JSON, or omit `--out` for CSV on stdout. Supply extracted descriptor tables with `--features features.csv`. Counting tables require the 16 radial measurements and `--assume-barrel`. Recognition and counting use separate annotation contexts, documented alongside the [models](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/docs/models.md).

### Checkpoints

The v1.1 release uses three version-pinned JSON checkpoints, distributed as release attachments separately from the software wheel. The first prediction installs a missing checkpoint into the local cache after checking its expected size and SHA-256 digest. Subsequent predictions reuse that verified copy. Inspect availability or pre-install all three:

```bash
betlas checkpoints list
betlas checkpoints install all
```

Use `--cache-dir` to choose the cache, `--checkpoint model.json` for an explicit local checkpoint, and `--offline` to require local files. A local release folder or mirror can populate the cache:

```bash
betlas checkpoints install all --base-url /path/to/release-assets --cache-dir /path/to/cache
betlas predict solenoid --structure protein.cif --chain A --cache-dir /path/to/cache --offline
```

For strand counting with explicit local files, supply `--barrel-checkpoint` for the recognition gate, or `--assume-barrel` for a known barrel. Inference reads numeric parameters directly from JSON. See the [model documentation](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/docs/models.md) for training populations, annotation procedures and prediction interpretation.

## Python API

The Python API supports the same descriptor extraction, slicing and prediction workflows:

```python
from betlas import extract_structure_features, predict_structure

features = extract_structure_features("outputs/example/mini.cif", chain_id="A")
print(features["betlas_beta_strand_count"])

result = predict_structure("barrel", "protein.cif", chain_id="A")
```

For an annotated `StructureGeometry`, use `betlas.extract_classifier_features(geometry)`. For measured inputs, use `betlas.predict_features(task, feature_mapping)`. All descriptor extraction paths share the current 136-field contract. Column aliases can be converted explicitly with `betlas.schema.normalize_feature_columns`.

## Benchmarks and related tools

The [CATH benchmark workflow](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/src/betlas/benchmarks/README.md) covers release catalogs, exact domain extraction, DSSP annotations, grouped splits and coverage-aware evaluation. [Specialist workflows](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/src/betlas/benchmarks/SPECIALISTS.md) compare task-specific methods and record their applicability.

The [geometry tools](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/src/betlas/geometry_relations/README.md) analyse packing, connectivity and structural correspondence. [Companion scripts](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/scripts/external_baselines/README.md) adapt separately installed methods and ESM-C embeddings. Source distributions include these scripts; the wheel installs the `betlas` package. Run `betlas COMMAND --help` for command options.

General assets use `BETLAS_ASSET_MANIFEST_DIR`, `BETLAS_ASSET_DIR` and `BETLAS_ASSET_BASE_URL` to select manifests, cache and source. Application checkpoints have their own built-in manifest and the installation interface described above.

## Development

```bash
python -m pip install '.[dev,geometry]'
python -m pytest
python -m build
```

Tests use independently constructed structures, tables and local tool stand-ins. Optional native alignment tests require USalign or TM-align. Application tests exercise numeric inference and checkpoint installation using local fixtures.

The [release guide](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/docs/releasing.md) covers package checks and publishing.

See the [changelog](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/CHANGELOG.md) for the v1.1 interface changes. Betlas uses the [MIT License](https://github.com/GeraltZeroZhong/Betlas/blob/v1.1.0/LICENSE). CATH, RCSB PDB, DSSP and external software retain their respective attribution and distribution terms.

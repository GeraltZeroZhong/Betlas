# Betlas

Betlas is a library-first framework for local beta-structure geometry
grammars, CATH-scale beta-fold benchmarks and secondary beta-barrel readouts.
It converts PDB/mmCIF domains into traceable geometry objects, extracts
interpretable topology features, benchmarks lightweight models and exposes
readouts for topology ambiguity, continuous fold organization, mixed topology,
beta-barrel detection and beta-barrel stave counts.

Betlas v1.0 is not a global fold-search replacement. Labels are external
CATH-derived source evidence, and Betlas readouts are reported as reproducible
geometry signals rather than ground-truth label revisions.

## Installation

```bash
conda env create -f environment.yml
conda activate betlas
pip install -e ".[ml,external,dev]"
```

The package exposes the `betlas` console script and can also be run from a
checkout with:

```bash
PYTHONPATH=src python -m betlas --help
```

## CLI

```bash
betlas build-dataset --all-eligible --out outputs/betlas_cath_labels.csv
betlas extract-features --labels outputs/betlas_cath_labels.csv --out outputs/betlas_cath_features.csv
betlas benchmark --features outputs/betlas_cath_features.csv --config configs/benchmark/full_scale.yaml
betlas ablate --features outputs/betlas_cath_features.csv --config configs/benchmark/full_scale_ablation.yaml
betlas readout list
betlas readout topology-diagnostics --config configs/readouts/topology_diagnostics_full.yaml
betlas publication-evidence --config configs/publication/evidence_full.yaml
betlas external-baselines --config configs/publication/external_baselines_full.yaml
```

The full pipeline wrapper remains:

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py --config configs/pipeline/full_cath.yaml
```

## Python API

```python
from betlas import extract_feature_row, count_stave_readout
from betlas.readouts.beta_barrel_detection import detect as detect_beta_barrel
from betlas.readouts.topology_diagnostics import compute_topology_diagnostics
```

The beta-barrel stave readout is available as a package API and CLI:

```bash
betlas readout beta-barrel-staves --help
```

The beta-barrel detection readout is available as:

```bash
betlas readout beta-barrel-detection --help
```

## Repository Layout

| Path | Public role |
| --- | --- |
| `src/betlas/` | Installable Betlas package, CLI and readout APIs |
| `configs/` | Reproducibility configuration files |
| `scripts/run_full_pipeline.py` | Top-level orchestration wrapper |
| `scripts/external_baselines/` | Public wrappers for external baseline tools |
| `reproducibility/` | Environment locks and reproducibility support files |
| `tests/` | Public unit and smoke tests for the tracked package code |

Generated documents, visual workspaces, release data packages, large local
data assets and long-run operation scripts are managed outside this code
repository and are excluded by `.gitignore`.

The v1.0 numerical artifact schema retains the existing `cz_` feature-column
prefix to preserve the fixed result contract. Public package names, commands,
schemas and documentation use Betlas.

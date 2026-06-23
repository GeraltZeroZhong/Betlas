# Readout Benchmark Reproduction Scripts

This directory contains repository companion scripts for fixed benchmark
cohorts and readout asset generation. The reusable readout implementations live
under `src/betlas/readouts`; scripts here orchestrate cohort construction,
feature-block assembly, supervised model settings, and summary exports.

## Contents

- `beta_barrel_detection/`: MPstruc-aligned beta-barrel detection cohort
  builders, Betlas 151 LayerRadial16 official runner, feature-block ablation
  runner, external-method alignment helper, and ESM-C embedding cache builder.
- `beta_barrel_staves/`: fixed-cohort Betlas 151 LayerRadial16 stave-count
  runner and asset-bundle input generator.

These scripts are tracked reproducibility material. They are not stable import
APIs and are not packaged into the PyPI wheel.

## Asset Workflow

Official input bundles generated here are represented by manifests under
`assets/` and by package resources consumed through `betlas assets ...`.
Large CSV/NPZ payloads should be distributed through a release bundle whose
layout matches the manifest `download_path` values. While packaged manifests
report `pending_release`, clean clones can run these scripts only with an
explicit local mirror or already-populated local asset cache; there is no
default public asset download path.

## Typical Usage

```bash
PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/beta_barrel_detection/run_betlas151_layer_radial16_official.py --help
PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/beta_barrel_staves/run_betlas151_layer_radial16_official.py --help
```

With a local mirror whose layout matches each manifest `download_path`:

```bash
export BETLAS_ASSET_BASE_URL=/mirror/betlas-assets
export BETLAS_ASSET_DIR=$PWD/runs/assets

PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/beta_barrel_detection/run_betlas151_layer_radial16_official.py \
  --download-assets \
  --out-dir runs/readouts/beta_barrel_detection_official

PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/beta_barrel_staves/run_betlas151_layer_radial16_official.py \
  --download-assets \
  --out-dir runs/readouts/beta_barrel_staves_official
```

After the assets are already cached:

```bash
betlas assets verify betlas-beta-barrel-detection-official-v1 --cache-dir runs/assets --strict
betlas assets verify betlas-beta-barrel-staves-official-v1 --cache-dir runs/assets --strict

PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/beta_barrel_detection/run_betlas151_layer_radial16_official.py \
  --asset-id betlas-beta-barrel-detection-official-v1 \
  --asset-cache-dir runs/assets \
  --out-dir runs/readouts/beta_barrel_detection_official

PYTHONPATH=.:src python scripts/reproducibility/readout_benchmarks/beta_barrel_staves/run_betlas151_layer_radial16_official.py \
  --asset-id betlas-beta-barrel-staves-official-v1 \
  --asset-cache-dir runs/assets \
  --out-dir runs/readouts/beta_barrel_staves_official
```

The runners also verify the manifest hash and byte-size contract for every
cached file they read, then record expected and observed file state in
`metadata.json`.

External comparison adapters live under `scripts/external_baselines/`; Betlas
core readouts can be invoked directly with `betlas readout ...`.

## Runtime Notes

Run commands from the repository root. The `PYTHONPATH=.:src` prefix makes both
the source package and companion script modules importable without installing
the repository as a package.

Fixed-cohort supervised runners require CatBoost and NumPy/Pandas. ESM-C
embedding builders require a local upstream ESM-C installation and local model
weights; Betlas asset helpers only resolve and verify those paths.

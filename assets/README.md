# Betlas Assets

This directory tracks asset manifests and checksums. Large payloads are mirrored
locally before release and are not stored in Git or in the PyPI package.

The Python package includes copies of these manifests so installed users can
query and verify assets without a source checkout:

```bash
betlas assets list
betlas assets describe betlas-beta-barrel-staves-official-v1
```

Current manifests are available before the large payloads are published. While
a manifest reports `pending_release`, downloads must use an explicit local
mirror whose layout matches each `download_path` value.

Configure local locations when needed:

- `BETLAS_ASSET_BASE_URL`: release URL or local directory containing the same
  relative paths as each manifest `download_path`
- `BETLAS_ASSET_DIR`: local cache directory, defaulting to
  `~/.cache/betlas/assets`
- `BETLAS_ESMC_WEIGHTS`: local ESM-C weights path for workflows that require
  the upstream model

Each manifest records the asset id, readout profile, generator command, file
names, SHA-256 checksums, byte sizes, release status, and release-relative
download paths. `pending_release` means the manifest is available but the
payload is not yet promised at the default release base.
The fixed-cohort detection and staves companion runners consume the cached
files named in these manifests; while assets are pending, they do not rebuild
official cohorts from hidden external directories unless the caller supplies
those inputs explicitly.

Download only after `betlas assets describe ...` reports a published payload or
after `BETLAS_ASSET_BASE_URL` points to a local mirror:

```bash
BETLAS_ASSET_BASE_URL=/path/to/betlas-assets \
  betlas assets download betlas-beta-barrel-staves-official-v1
betlas assets verify betlas-beta-barrel-staves-official-v1 --strict
```

Downloads are written atomically and verified before replacing any cached file.

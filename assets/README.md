# Betlas Assets

This directory tracks asset manifests and checksums. Large payloads are released
as external asset bundles and are not stored in Git or in the PyPI package.

The Python package includes copies of these manifests so installed users can
query and verify assets without a source checkout:

```bash
betlas assets list
betlas assets describe betlas-beta-barrel-staves-official-v1
```

Current official manifests point at release zip bundles. Betlas can download
from the default release URL, or from a local mirror that provides the same zip
files or an unpacked directory matching each `download_path` value.

Configure local locations when needed:

- `BETLAS_ASSET_BASE_URL`: release URL or local directory containing the
  manifest bundle zip files or the same relative paths as each manifest
  `download_path`
- `BETLAS_ASSET_DIR`: local cache directory, defaulting to
  `~/.cache/betlas/assets`
- `BETLAS_ESMC_WEIGHTS`: local ESM-C weights path for workflows that require
  the upstream model

Each manifest records the asset id, readout profile, generator command, bundle
name, file names, SHA-256 checksums, byte sizes, release status, and
release-relative download paths.
The fixed-cohort detection and staves companion runners consume the cached
files named in these manifests.

Download from the default release URL:

```bash
betlas assets download betlas-beta-barrel-staves-official-v1
betlas assets verify betlas-beta-barrel-staves-official-v1 --strict
```

For an offline mirror:

```bash
BETLAS_ASSET_BASE_URL=/path/to/betlas-assets \
  betlas assets download betlas-beta-barrel-staves-official-v1
```

Downloads are written atomically and verified before replacing any cached file.

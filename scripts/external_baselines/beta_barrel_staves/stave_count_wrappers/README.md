# External Baselines

This directory contains wrappers for beta-barrel baseline methods that can be
evaluated through recorded wrappers from public author/original code or packaged original software. The
wrappers do not reimplement the methods; they only fetch, build, and call the
upstream tools.

## Implemented Wrappers

| Method | Upstream | Local wrapper | Scope |
| --- | --- | --- | --- |
| PROFtmb | Debian/Ubuntu package of the original C++ PROFtmb software | `scripts/run_proftmb.sh` | Profile-HMM runner |
| BETAWARE | `https://github.com/BolognaBiocomp/betaware` | `scripts/run_betaware.sh` | Profile-based runner |
| PRED-TMBB2 | `https://github.com/pbagos/juchmme` release 1.0.6 | `scripts/run_juchmme_pred_tmbb2.sh hmm` | HMM topology runner |
| PRED-TMBB2+HNN | JUCHMME HNN config/weights | `scripts/run_juchmme_pred_tmbb2.sh hnn` | HNN topology runner |
| TMbed | `https://github.com/BernhoferM/TMbed` | `scripts/run_tmbed.sh` | Sequence/embedding runner |
| PolarBearal3 | `https://github.com/SluskyLab/PolarBearal3` | `scripts/run_polarbearal.sh` | Structure runner |

## Pinned Sources

The fetch script checks out fixed upstream revisions where the upstream project
does not publish immutable releases:

| Method | Pinned source |
| --- | --- |
| BETAWARE | `BolognaBiocomp/betaware@c2797d595798c2febd30bacbde73222a01c4119b` |
| TMbed | `BernhoferM/TMbed@8cee893523eb655bc9485c00c65336d27a236191` |
| PolarBearal3 | `SluskyLab/PolarBearal3@e8476c3d8bed6c07bb0f0b7e604be183eafe3a15` |
| PRED-TMBB2 / HNN | JUCHMME `1.0.6` release zip |
| PROFtmb | Debian/Ubuntu `proftmb_1.1.12-11_amd64.deb` package as tested |

## Layout

- `scripts/`: tracked wrapper/setup code.
- `sources.json`: reproducibility decision for each method in the catalog.
- `tools/`: ignored local cache for cloned/downloaded upstream tools.
- `.conda/`: ignored local environments for Mono and TMbed.
- `runs/`: ignored generated smoke-test outputs.

Only wrapper code and method metadata are tracked. Third-party source trees,
local conda environments, generated predictions, downloaded model weights,
structure data and local notes are excluded by `.gitignore`.

## Setup

Fetch the upstream tools:

```bash
bash scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/fetch_external_tools.sh
```

Create the optional local environments used by PolarBearal3 and TMbed:

```bash
bash scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/setup_envs.sh
```

Run all smoke tests:

```bash
bash scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_smoke_tests.sh
```

The smoke tests use upstream example inputs when available. PolarBearal3 uses an
RCSB PDB-format download of 1A0S because the upstream C# parser reads PDB fixed
columns, not mmCIF.

## Notes

PolarBearal3 is built with .NET SDK 6 from the official C# source. The wrapper
adds a small CLI entry point that calls the upstream `ReadPdbFile`,
`MonoProtein`, `MonoBarrel`, and `PolarBearal` classes. The current upstream
main branch also needs a compile-only missing `break;` fix in one unused menu
branch; the wrapper applies that before build and does not change the strand
assignment code path.

For NMR PDB entries, the gold runner keeps only the first `MODEL ... ENDMDL`
block before chain filtering. PolarBearal3's own parser stops at `ENDMDL`; this
preserves that intended behavior and avoids counting all NMR models as one
chain.

Gold dataset runs for profile-based methods are reproducible, but the bundled
default creates PSI-BLAST profiles from the local gold FASTA only. That is useful
for auditing wrapper execution and file formats; it is not a complete
NR/UniRef-style profile benchmark. For a full metric comparison, set up and
document a frozen external BLAST database and rerun the same wrappers.

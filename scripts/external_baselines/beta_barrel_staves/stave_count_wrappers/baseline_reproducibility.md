# Stave-Count External Baseline Reproduction

This note records the external stave-count methods covered by the wrapper
collection and the reproducibility assumptions for each method. Wrappers call
author code, official source repositories, or packaged software. They do not
reimplement missing algorithms.

## Integrated Wrappers

| Method | Upstream code/package | Wrapper behavior |
| --- | --- | --- |
| PROFtmb | Debian/Ubuntu `proftmb` package | Runs the packaged profile-HMM binary on PSI-BLAST ASCII PSSM profiles. |
| PRED-TMBB2 | JUCHMME release 1.0.6 | Calls the Java HMM implementation with `tmbb2.mdel`, transition/emission tables, and `conf.tmbb`. |
| TMbed | `https://github.com/BernhoferM/TMbed` | Runs the upstream Python package with optional precomputed embeddings. |
| PolarBearal3 | `https://github.com/SluskyLab/PolarBearal3` | Builds the C# source with .NET SDK 6 and calls the upstream barrel-analysis classes through a small CLI shim. |

## Available but Not Used as Primary Metrics

| Method | Reason |
| --- | --- |
| BETAWARE | Requires MSA frequency profiles built from a large non-redundant sequence database; local self-FASTA profiles are useful only for wrapper smoke checks. |
| PRED-TMBB2+HNN | The local HNN invocation completes but fails bundled-barrel sanity checks in the FASTA decoding path; the HMM mode is the reproducible JUCHMME path used here. |

## Pinned Versions

| Method | Version or revision |
| --- | --- |
| PROFtmb | `proftmb_1.1.12-11_amd64.deb` |
| BETAWARE | `BolognaBiocomp/betaware@c2797d595798c2febd30bacbde73222a01c4119b` |
| PRED-TMBB2 / HNN | JUCHMME `1.0.6` release zip |
| TMbed | `BernhoferM/TMbed@8cee893523eb655bc9485c00c65336d27a236191` |
| PolarBearal3 | `SluskyLab/PolarBearal3@e8476c3d8bed6c07bb0f0b7e604be183eafe3a15` |

## Running Smoke Checks

```bash
bash scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/fetch_external_tools.sh
bash scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/setup_envs.sh
bash scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_smoke_tests.sh
```

Generated tool checkouts, local environments, downloaded files, prediction
outputs, model weights, and run directories are excluded from source control.

## Profile-Based Methods

PROFtmb and BETAWARE require profile inputs. The wrappers can exercise file
formats with small local profiles, but metric-quality runs should use a frozen
external profile database and record the database version, checksum, and profile
generation command.

## Structure-Based Methods

PolarBearal3 consumes fixed-column PDB records. The wrapper can filter a single
chain into a PDB file before invoking the upstream parser. For NMR entries, only
the first model block should be used so one ensemble is not concatenated into a
single artificial chain.

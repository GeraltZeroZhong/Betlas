# External Baseline Wrappers

This directory contains wrappers for running external baseline methods against
Betlas readout inputs. The wrappers do not vendor upstream tools, model weights
or databases. They provide command-line adapters, input normalization, output
parsing and denominator-aligned summaries while keeping third-party runtime
requirements outside the core package.

## Scope

- `beta_barrel_detection/`: adapters for whole-chain beta-barrel detection
  baselines.
- `beta_barrel_staves/stave_count_wrappers/`: wrappers for beta-barrel
  transmembrane-strand or stave-count baselines.

Large downloaded databases, tool checkouts, temporary work directories and run
outputs are intentionally excluded from the source tree. Install each upstream
method under the license and citation requirements of that project, then pass
its executable or checkout path to the corresponding wrapper.

## External Methods

| Method | Role in Betlas analyses | Upstream source |
| --- | --- | --- |
| Foldseek | Structure-search baseline for beta-barrel detection and external structural context | <https://github.com/steineggerlab/foldseek> |
| IsItABarrel | Structure/contact-map beta-barrel detection baseline | <https://github.com/SluskyLab/isitabarrel> |
| PRED-TMBB2 / JUCHMME | Sequence-based transmembrane beta-barrel topology baseline | <https://github.com/pbagos/juchmme> |
| BetAware | Transmembrane beta-barrel detection/topology baseline | <https://github.com/BolognaBiocomp/betaware> |
| TMbed | Transmembrane-protein prediction baseline used for aligned stave-count comparisons | <https://github.com/BernhoferM/TMbed> |
| PROFtmb | Profile-HMM transmembrane beta-barrel baseline | <https://packages.debian.org/bullseye/proftmb> |
| PolarBearal3 | Beta-barrel strand-count baseline wrapper | <https://github.com/SluskyLab/PolarBearal3> |

## Reproducibility Contract

The wrappers preserve three boundaries.

1. External tools are not used to define Betlas labels, features or trained
   readouts.
2. Missing external predictions are retained as missing or denominator-specific
   outcomes rather than silently dropped.
3. Wrapper outputs are post hoc comparison tables. They are not portable
   replacements for the upstream software, and they do not modify Betlas run
   artifacts.

When adding a new external baseline, include a method-specific README with the
upstream URL, version or commit policy, expected input files, generated output
schema and any license constraints that affect redistribution.

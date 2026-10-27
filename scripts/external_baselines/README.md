# External methods

These source-checkout adapters expose external methods used alongside Betlas.
They accept caller-supplied inputs, local tool installations, and output paths.
Install Betlas for shared Python dependencies, then follow the method-specific
instructions for the external program and its models.

| Function | Methods | Guide |
| --- | --- | --- |
| Beta-barrel detection | Cooper-Beta, Foldseek/TMalign, IsItABarrel, PRED-TMBB2 through JUCHMME | [Detection](beta_barrel_detection/README.md) |
| Specialist binary tasks | Shared labelled-reference Foldseek and native method evaluation | [Task guide](SPECIALIST_TASKS.md) |
| Beta-barrel strand counts | BETAWARE, PRED-TMBB2 HMM and HNN, PROFtmb, TMbed, PolarBearal3 | [Strand counts](beta_barrel_staves/README.md) |
| Protein embeddings | ESM-C 600M | [Embeddings](embeddings/README.md) |
| Protein sequence search | NCBI BLASTp, MMseqs2 | [Sequence search](sequence_search/README.md) |
| Solenoid detection | RAPHAEL output import, SOLeNNoID | [Solenoids](solenoid/README.md) |
| Sequence-based beta pairing and contacts | BetaPro, BCov, bbcontacts native adapters | [Beta topology](beta_topology/README.md) |

The guides describe executable dependencies, output schemas, parsing rules, and
upstream attribution. ProtT5 is a TMbed dependency; its inference is managed by
that upstream program. Install each upstream implementation and obtain its weights
under the applicable license and citation terms.

Specialist outputs have different endpoints and information requirements. Follow
the [comparison contracts](../../src/betlas/benchmarks/SPECIALISTS.md) before
evaluating solenoid scores, strand partners or residue contacts. Check that the
required upstream runtime, MSA pipeline or web service is available. Select methods
whose endpoints and information requirements fit the planned CATH comparison.

Run Python entry points from the checkout root, for example:

```bash
python -m scripts.external_baselines.beta_barrel_detection.foldseek.runner --help
python -m scripts.external_baselines.embeddings.esmc --help
python -m scripts.external_baselines.sequence_search --help
bash scripts/external_baselines/beta_barrel_staves/run_tmbed.sh --help
```

The [strand-count example](beta_barrel_staves/README.md#synthetic-parsing-example)
demonstrates an end-to-end conversion with invented sequence labels. The
independently constructed [mini structure](../../examples/mini.cif) supports input
preparation tests. Detection guides also show how to supply positive and negative
directories for a caller-defined evaluation.

Place inputs under `inputs/` and generated files under `outputs/`, or select
directories outside the checkout. Run one job per output directory when a method
uses fixed sidecar names. The source distribution includes these scripts; the
Python wheel installs the core `betlas` package.

## Verification and contributions

```bash
python -m pytest tests/test_external_detection.py tests/test_external_staves.py \
  tests/test_external_embeddings.py tests/test_external_sequence_search.py
```

Synthetic parser tests, analytical expectations, and temporary fake executables
check preparation, argument passing, failure propagation, and output alignment.
Model inference and biological accuracy require the separately installed methods
and suitable inputs. Add adapters by function, preserve explicit paths, and include
tests that run with synthetic inputs and locally constructed command substitutes.

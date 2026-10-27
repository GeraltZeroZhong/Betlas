# External beta-barrel strand-count methods

These adapters run locally installed external methods on explicit inputs and
convert their outputs to a shared strand-count table. Each external program,
its models, and its runtime environment are supplied separately. The scripts
require Bash on Linux or WSL; Python utilities require Python 3.10 or later.

| Method | Adapter | Required local installation | Count interpretation |
| --- | --- | --- | --- |
| BETAWARE | `run_betaware.sh` | Python and `BETAWARE_ROOT` | Reported topology intervals |
| PRED-TMBB2 HMM | `run_juchmme_pred_tmbb2.sh hmm` | Java and `JUCHMME_ROOT` | Contiguous `M` runs in `VP` labels |
| PRED-TMBB2 HNN | `run_juchmme_pred_tmbb2.sh hnn` | Java and `JUCHMME_ROOT` | Contiguous `M` runs in `VP` labels |
| PROFtmb | `run_proftmb.sh` | `PROFTMB_ROOT` with executable, data, and runtime libraries | `#_Predicted_Strands` column |
| TMbed | `run_tmbed.sh` | Python environment, `TMBED_ROOT`, and local model files | Contiguous `B`/`b` runs in format 1 topology |
| PolarBearal3 | `run_polarbearal.sh` | .NET 6, `POLARBEARAL_ROOT`, and `POLARBEARAL_DLL` | Strand count from the CLI shim |

Run any adapter with `--help` to see its input contract and environment variables.
Help works before installing external methods. Set each tool root to an absolute
local directory and select a method-specific interpreter where needed. The
wrappers use explicit output paths. Each wrapper prints its result path. TMbed
and PROFtmb can also write upstream progress messages to stdout; other adapters
capture method output in the documented result or log file.

## Synthetic parsing example

This example uses an invented eight-residue sequence and independently constructed
labels with two beta segments to demonstrate output conversion:

```bash
mkdir -p /tmp/stave-example
printf '>toy\nACGTACGT\niiBBooBb\n' > /tmp/stave-example/toy.pred
python scripts/external_baselines/beta_barrel_staves/parse_counts.py \
  --method tmbed --input /tmp/stave-example/toy.pred \
  --output /tmp/stave-example/counts.tsv
```

The TSV columns are `record_id`, `method`, `strand_count`, `status`, and `raw`.
The toy count is `2`. `status=failed` and a blank count identify a recognized
record without a usable prediction; a valid zero remains `0`. Empty or unsupported
files fail with an error. `raw` retains topology or summary text to make the
conversion inspectable. Both JUCHMME modes use the same output parser. BETAWARE
parsing requires `--record-id` because the adapter consumes one result at a time.

## Local execution

Examples below use caller-supplied files and installations:

```bash
export BETAWARE_ROOT=/path/to/betaware
bash scripts/external_baselines/beta_barrel_staves/run_betaware.sh input.fasta input.profile output.txt

export JUCHMME_ROOT=/path/to/juchmme_git
bash scripts/external_baselines/beta_barrel_staves/run_juchmme_pred_tmbb2.sh hmm input.fasta output.txt
bash scripts/external_baselines/beta_barrel_staves/run_juchmme_pred_tmbb2.sh hnn input.fasta output-hnn.txt

export PROFTMB_ROOT=/path/to/unpacked-installation
bash scripts/external_baselines/beta_barrel_staves/run_proftmb.sh input.Q output/result

export TMBED_ROOT=/path/to/TMbed
export TMBED_PYTHON=/path/to/environment/bin/python
bash scripts/external_baselines/beta_barrel_staves/run_tmbed.sh input.fasta embeddings.h5 output.pred
```

BETAWARE consumes a frequency profile. `profile_from_pssm.py input.pssm
output.profile` converts the twenty observed-percentage columns of a PSI-BLAST
ASCII PSSM to fractions, retaining the original two-decimal formatting. Set
`BETAWARE_AA_ORDER=ARNDCQEGHILKMFPSTWYV` for this profile order. Profiles must match
the query sequence and the method's preprocessing assumptions. The wrapper keeps
the upstream profile order when this override is omitted and exposes the existing
sensitivity option through `BETAWARE_THRESHOLD` (default `0.8`).

PROFtmb accepts a query profile file or profile directory in its own query format;
the `PROFTMB_*` overrides also support other local installation layouts. TMbed
runs on CPU. It can reuse embeddings or receive `-` to create embeddings with
already available language-model weights. Hugging Face offline flags prevent
automatic model fetches through that client; install the required weights first.
These application settings control the Hugging Face client; network isolation
requires a separate sandbox.

### PolarBearal3 shim

The shim calls the original `MonoProtein`, `MonoBarrel`, and `PolarBearal` code
and analyzes the first chain in the input PDB. Supply a single-chain structure
when comparing chain-level counts. The summary is tab-separated: structure name,
strand count, axis length, mean radius, mean tilt, and shear number. Upstream
diagnostic text may precede this line.

```bash
# --restore explicitly permits NuGet dependency restoration.
bash scripts/external_baselines/beta_barrel_staves/build_polarbearal.sh \
  --restore /path/to/PolarBearal3 /path/to/build
export POLARBEARAL_ROOT=/path/to/PolarBearal3
export POLARBEARAL_DLL=/path/to/build/bin/PolarBearal3Cli.dll
bash scripts/external_baselines/beta_barrel_staves/run_polarbearal.sh input.pdb output
```

The build uses .NET SDK 6 and MathNet.Numerics 5.0.0. Subsequent builds omit
`--restore` to use existing restored dependencies. The build script copies the
upstream C# files into the chosen build directory and applies the known missing
`break` fix in the menu switch to that copy. The algorithm implementation and
supplied checkout remain unchanged. Use a separate build directory for each
upstream checkout.

## Upstream attribution and versions

The [source manifest](sources.json) records the upstream software references used
by these adapters. BETAWARE is provided by
[BolognaBiocomp](https://github.com/BolognaBiocomp/betaware); JUCHMME/PRED-TMBB2 by
[pbagos](https://github.com/pbagos/juchmme); PROFtmb is available as a
[Debian package](https://packages.debian.org/bullseye/proftmb); TMbed by
[BernhoferM](https://github.com/BernhoferM/TMbed); and PolarBearal3 by
[SluskyLab](https://github.com/SluskyLab/PolarBearal3).
Obtain each program from its upstream distribution and retain its license and
citation notices. This directory contains integration code; upstream code,
trained weights, and reference data are obtained separately under their respective
terms. Method-specific predictive assumptions remain those of the upstream tools.

To contribute an adapter, preserve explicit input/output paths, document the
upstream output schema, and add independently constructed parser or command
contract tests. Run `python -m pytest tests/test_external_staves.py` for the
offline synthetic checks. These checks validate integration behavior; scientific
accuracy requires separate validation on an appropriate dataset.

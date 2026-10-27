# Beta-residue contacts and strand connectivity

These optional adapters expose BetaPro, BCov, and bbcontacts for residue contacts
and strand connectivity. CATH architecture and topology classification require
a separate classifier. Sequence/MSA predictions and coordinate-derived graphs
use different inputs; account for that difference in comparisons. Independent
validation of a DSSP-derived Betlas graph requires a reference independent of
DSSP. Use a contact/pairing protocol when that endpoint is needed.

Supply compatible upstream programs, weights, research inputs, and biological
reference outputs separately. Public tests use synthetic inputs.

## Native contracts

| Method | Required inputs | Native outputs | Adapter coverage |
| --- | --- | --- | --- |
| BetaPro 1.0 | FASTA plus SSpro4 profiles/accessibility; optionally supplied SS; or an existing native probability map | `.map` dense probabilities over beta residues; `.str` strand pairs, alignment, orientation, pseudo-energy | Import both formats; run separately configured legacy scripts with a total timeout |
| BCov 1.0 | Ungapped target-first PSICOV-format MSA **or** sparse residue coupling scores; strand intervals | `.topo` strand graph; `.bmap` hard residue pairs | Validate inputs, invoke native options, import outputs |
| bbcontacts | Dense CCMpred coupling matrix; MSA row count; H/E/C predicted or observed SS | Sparse filtered residue pairs, orientation, path-level Viterbi scores | Validate dimensions, derive `sqrt(N)/L`, run native CLI, import results |

BCov source `src/main.c` writes two columns to `.bmap`: zero-based residue
indices. The three-column matrix in its README describes input scores. The
adapter preserves BCov hard predictions with confidence scores left unavailable.
Strand input `.ss` has a strand count followed by
zero-based `strand_id start end`; ends are inclusive. `.topo` contains the same
count and `strand_i strand_j direction`, where `0=parallel`, `1=antiparallel`,
and `2=isolated bridge`.

bbcontacts filtered output has eight whitespace-separated columns:

```text
#identifier diversity direction viterbiscore indexpred state res1 res2
toy 0.40 Antiparallel -2.1 1 first 8 2
toy 0.40 Antiparallel -2.1 1 last 7 3
```

Residues are already one-based. Each score is an uncalibrated value for a
predicted path; the output contains only the reported residue pairs.
Native first/internal/last and all six bulge states are preserved; upstream
evaluates the emitted bulge-state pairs as contacts too.
Upstream emits an all-`NA` row when no path remains and before some errors.
The importer therefore requires independently verified successful-exit evidence.
Empty files, header-only files, duplicate pairs, and mixed/appended empty outputs
fail validation. Every runner uses a fresh output directory because upstream
bbcontacts appends to existing outputs.

BetaPro `.map` begins with name, full sequence, secondary structure, and two
unused partner lines; subsequent rows form a symmetric matrix indexed by E
residues in sequence order. Map these indices to full-sequence positions before
comparison. Both three-state SS and the non-E DSSP symbols in the official
supplied-SS example are accepted. `B` as a distinct SS state is unsupported
because the inspected examples leave its index convention unresolved; declare
a conversion to the upstream E-as-strand/bridge convention if needed.
The `.str` parser keeps one-based sequence intervals, expands aligned residue
pairs, and validates parallel/antiparallel order and strand coverage. BetaPro's
`A/P/B` pairing directions mean antiparallel/parallel/bridge. Pseudo-energy and
probability are stored separately.

## Normalized JSON

Each result has `schema_version`, `tool`, `sample_id`, `sequence_length`,
`native_index_base`, `normalized_index_base=1`, `status`, `contacts`,
`strand_pairs`, and `strands`. Status is `ok` or a verified `empty`. Launch,
timeout, process-exit, and parsing failures raise exceptions; the success file
is written only after successful execution and parsing.

| Array | Fields |
| --- | --- |
| `contacts` | `residue_i`, `residue_j`, `orientation`, `score`, `score_kind` |
| `strand_pairs` | `strand_i`, `strand_j`, `orientation`, `score`, `score_kind` |
| `strands` | `strand_id`, `start`, `end` (inclusive sequence positions) |
| `score_matrix` (BetaPro only) | Full-sequence residue indices and `residue_pair_probability`, independently of selected contacts |

Pair members are sorted. Strand IDs are local to each supplied/predicted strand
definition; they must be aligned to a declared reference before comparing graphs.
The sequence-position-to-author-residue map, gaps, insertion codes, and domain
segments belong to the caller's benchmark contract. Use that mapping to join
sequence positions to PDB residue numbers.

Runners add `secondary_structure_source` (`predicted` or `observed`) and explicit
input provenance. The omitted-pair statement describes sparse native output.
Define eligible pairs, negative labels and the evaluation denominator separately.
Compute ranking metrics only where genuine scores cover the declared universe.
BCov supplies hard contacts with confidence scores left unavailable; use metrics
defined for binary calls.

## Use

```bash
python -m scripts.external_baselines.beta_topology --help
python -m scripts.external_baselines.beta_topology parse bbcontacts \
  --input inputs/toy.filteredoutput.txt --sample-id toy --length 10 \
  --successful-exit --output outputs/toy.json
python -m scripts.external_baselines.beta_topology run bcov --spec inputs/job.json
```

The JSON run specification uses the exact named arguments of
[`runner.py`](runner.py). For example:

```json
{
  "command": ["/path/to/bcov"],
  "sample_id": "toy",
  "sequence": "ACDEFG",
  "strands_file": "/path/to/toy.ss",
  "scores": "/path/to/toy.mat",
  "ss_source": "observed",
  "output_dir": "/path/to/new-run-directory",
  "timeout": 300
}
```

The corresponding synthetic `.ss` and sparse input matrix are:

```text
2
0 0 1
1 4 5
```

```text
6
0 4 0.7
1 5 0.6
```

`run_bcov` accepts exactly one of `msa` and `scores`, defaults to the upstream
parallel-strand minimum separation of six, and requires at least two strands
because the native score normalization divides by the strand-pair count.
Substituting non-PSICOV couplings is a separately named method variant.

`run_bbcontacts` takes `command`, `sample_id`, `sequence`, `coupling_matrix`,
`secondary_structure`, `ss_source`, `alignment_sequence_count`, `output_dir`,
`timeout`, and optional `config`. Its matrix must be tab-delimited and square;
the FASTA-like SS file contains H/E/C states only. `alignment_sequence_count`
is the actual number of input alignment rows, including redundant sequences.
The unchanged upstream defaults use smoothing range 10 and prediction
shortening. Upstream ignores couplings below diversity 0.022, caps diversity at
1.98, and changes Viterbi time-saving thresholds for long sequences; retain
upstream logs and configuration when interpreting such cases.

`run_betapro` takes `installation`, `sample_id`, `sequence`, `input_file`,
`input_kind` (`fasta`, `supplied_ss`, `probability_map`), `ss_source`,
`output_dir`, and `timeout`. It runs the verified `predict_beta_fasta.sh` or
`predict_beta_ss.sh` followed by `predict_strand.sh`. It rejects untouched
historical installation paths. Supplied SS also requires SSpro4 for profiles
and accessibility. The original unquoted shell/Perl paths require installation
and job paths without spaces or shell metacharacters. Inputs are staged under
simple names, protecting source files from native sidecars. The runner catches
the legacy strand wrapper's printed predictor failure even if its exit is zero.

Every stage saves command, elapsed time, exit/timeout state, stdout, and stderr.
Timeouts kill and reap the owned local process group. Execution stays local.
A successful run is parsed before `normalized.json` is written.

## Feasibility and cost

bbcontacts needs NumPy/SciPy and was written for Python 2.6/2.7/3.4. Check
compatibility on a small input when using a newer Python version.
Its dense matrix and dynamic-programming state scale with the square of sequence
length, while repeated paths/shortening affect runtime. MSA search and coupling
generation are separate, often dominant costs. The original pipeline expects
CCMpred default couplings. Other contact matrices define a different input
variant and require separate evaluation.

BCov requires C/Fortran compilation and GLPK. Its MSA path runs PSICOV internally:
the inspected source holds five float64 `(21L) × (21L)` matrices at once, around
8.35 GiB at `L=713`, before other arrays and Fortran workspace. Size concurrent
jobs from peak memory requirements.
The supplied-score path avoids this covariance stage but still incurs alignment
and integer-programming cost.

BetaPro's distributed binaries are 32-bit Linux executables requiring legacy
`libstdc++.so.5`, configured paths, model files, and SSpro4. Its native inference
and upstream preprocessing need validation before estimating full-run wall time.
The adapter timeout caps execution. Estimate runtime from representative
biological inputs across the intended length range.

## Sources and licenses

- [BetaPro author page](https://ics.uci.edu/~baldig/betasheet_data.html),
  [distribution](https://download.igb.uci.edu/betapro-1.0.tar.gz),
  [paper](https://doi.org/10.1093/bioinformatics/bti1004). The distribution allows
  scientific/academic use and restricts further redistribution without permission.
- [BCov author distribution](https://biocomp.unibo.it/savojard/bcov/bcov-1.0.tar.gz),
  [paper](https://doi.org/10.1093/bioinformatics/btt555). GPLv3; consult bundled
  notices for PSICOV, graphical lasso, and GLPK dependencies.
- [bbcontacts author repository](https://github.com/soedinglab/bbcontacts),
  [paper](https://doi.org/10.1093/bioinformatics/btv041). AGPLv3 or later.

```bash
python -m pytest tests/test_external_beta_topology.py
```

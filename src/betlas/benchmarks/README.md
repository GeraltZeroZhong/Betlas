# Frozen CATH domain benchmarks

This workflow preserves the native CATH architecture or topology labels from a
fixed release. All commands use local inputs and explicit output directories.
Provide classification files, compressed PDB mmCIF files and any model weights
locally.

## Catalog and representative selection

```bash
python -m betlas.benchmarks.preparation catalog \
  --domain-list /data/cath/cath-domain-list.txt \
  --boundaries /data/cath/cath-domain-boundaries.txt \
  --names /data/cath/cath-names.txt \
  --official-representatives /data/cath/cath-domain-list-S35.txt \
  --coordinate-index /data/pdb/acquisition.jsonl \
  --out-dir /data/project/catalog
```

The coordinate index is JSONL with `pdb_id`, `path`, `state`, and `sha256` fields.
Paths may be absolute or relative to the index. Successful states are
`downloaded`, `reused_byte_verified`, `reused_existing_collection`, and `ok`.
Content identities detect changed coordinate files when extraction resumes.
CATH domain IDs and chain IDs retain their original case.

`domains.csv` retains every selected release domain, exact ordered boundary
segments, native hierarchy and complete S35/S60/S95/S100 cluster identifiers.
`representatives.csv` contains one official S35 representative per cluster.
Replacement is allowed only for known source unavailability before feature
extraction, using resolution, length and domain ID as deterministic priorities.
A cluster with no available member retains its original representative and an
unavailable status. Keep the chosen representative when geometry extraction or
classification fails. Prior use is `unknown` unless the caller supplies a
complete domain-ID list with `--previously-used`.

## Freeze partitions before extracting features

```bash
python -m betlas.benchmarks.classification challenge \
  --manifest /data/project/catalog/representatives.csv \
  --grouping-manifest /data/project/catalog/domains.csv \
  --max-homologies 4 --seed 13 --out-dir /data/project/splits/rare-topology

python -m betlas.benchmarks.classification split \
  --manifest /data/project/catalog/representatives.csv \
  --grouping-manifest /data/project/catalog/domains.csv \
  --development-from /data/project/splits/rare-topology/splits.csv \
  --label-column cath_architecture_code --group-column cath_homology_code --pdb-policy purge \
  --n-splits 5 --seed 13 --out-dir /data/project/splits/architecture
```

Freeze the challenge first. Development splits reserve its held-out and purged
records; use these development splits for feature, model and weighting choices.
Whole-cohort CV without `--development-from` provides a separate assessment after
all choices for the challenge have been frozen.

The primary `--pdb-policy purge` protocol assigns whole hierarchy groups to
folds, then removes training records sharing any PDB with the full-catalog
held-out groups. Training support is checked again after this purge. The
`connected` alternative joins domains sharing the hierarchy key **or** a PDB
entry and preserves bridges through unselected domains. Large connected
components can make this alternative unsuitable as a balanced primary protocol;
report their sizes and retain it as a distinct sensitivity analysis. For topology
recognition use `--label-column cath_topology_code` with homology grouping. Report
S35 grouping separately because it measures within-family generalization.
Classes need at least two independent components to contribute to a closed-set
test in some folds. `support.csv` and long-form `splits.csv` retain unavailable
classes and reasons. The rare-topology challenge holds out topologies while
retaining architecture support and purges shared PDB entries.

## Uniform secondary structure and exact domain extraction

```bash
python -m betlas.benchmarks.preparation extract \
  --manifest /data/project/catalog/representatives.csv \
  --dssp-binary /tools/bin/mkdssp \
  --workers 2 --model-id 0 --limit-pdb 4 \
  --out-dir /data/project/features

# Continue the same frozen extraction after inspecting the bounded pilot:
python -m betlas.benchmarks.preparation extract \
  --manifest /data/project/catalog/representatives.csv \
  --dssp-binary /tools/bin/mkdssp \
  --workers 2 --model-id 0 --resume \
  --out-dir /data/project/features
```

DSSP annotates the selected full coordinate model once per PDB. Exact domain
selection follows author chains, insertion codes, ordered CATH segments and
polymer sequence positions. Discontinuous segments remain separate. Missing or
ambiguous boundaries produce a failure status. Use a DSSP version producing
annotated mmCIF strand/ladder categories (validated with 4.6.1). This extraction
uses DSSP annotations throughout. DSSP adjacency supports branched/cyclic sheets;
linear sheet-order descriptors remain missing when a unique order is unavailable.

A SQLite checkpoint stores result rows and compressed annotations atomically per
structure. Resume rejects changed input manifests, source implementations,
environments or DSSP versions. `--retry-failed --resume` retries failed rows;
annotation caching avoids repeating successful annotation. `features.csv` keeps
all input records, including unavailable, failed and pending domains, and
`extraction.json` distinguishes completion from successful coverage. Sequence
columns describe the selected observed domain residues. Multichain domains
expose `chain_sequences_json` and leave `domain_sequence` empty until an explicit
per-chain pooling policy is selected, preserving chain boundaries for sequence
embedding.

For an independent structure-search baseline, export exact full-atom domains
directly from original coordinates. This path uses the same model and boundary
rules and runs independently of DSSP:

```bash
python -m betlas.benchmarks.coordinate_exports \
  --manifest /data/project/catalog/representatives.csv \
  --model-id 0 --out-dir /data/project/domain-structures
```

The raw export command supports `--limit-domains` and `--resume`; its checkpoint
validates source and output content before reuse. The separate
`preparation export` command remains available for exact exports from a compatible
DSSP checkpoint when that dependency is intentional.

`domain_structures.csv` maps native IDs to hex-encoded filenames, preserving
case distinctions on case-insensitive filesystems. Exports contain the cropped
atom sites and omit full-structure sheet annotations. Build each Foldseek
reference database only from that split's training domain exports.

## Paired CPU model comparisons

```bash
python -m betlas.benchmarks.classification run \
  --manifest /data/project/catalog/representatives.csv \
  --features /data/project/features/features.csv \
  --splits /data/project/splits/architecture/splits.csv \
  --model logistic --weighting none --seed 13 \
  --out-dir /data/project/runs/logistic-unweighted
```

For a sequence baseline independent of DSSP success, join the original fixed
CATH domain FASTA to the same representative catalog:

```bash
python -m betlas.benchmarks.sequences \
  --manifest /data/project/catalog/representatives.csv \
  --fasta /data/cath/cath-domain-seqs.fa \
  --out-csv /data/project/sequences.csv
```

These published domain sequences may contain residues absent from observed
coordinates. Record that modality difference; the alternative observed-domain
sequence from feature extraction remains dependent on DSSP/domain mapping.
Use the explicit sequence mode of the [ESM-C adapter](../../../scripts/external_baselines/embeddings/README.md)
and the same saved classification splits.

Supported heads are logistic regression and random forest. Weighting choices are
`none`, `inverse`, `sqrt`, `effective`, and `group-balanced`; compare them on the
same saved splits. Group balancing defaults to H (`--weight-group-column
cath_homology_code`) to balance family contributions within each architecture. The
explicit S35 alternative becomes identical to inverse class weighting when
there is exactly one representative per S35 cluster. Preprocessing and weights
are fitted on training rows only.
Defaults select raw geometric features, excluding rule scores and metadata.
Use `--feature-columns col1,col2,...` for an explicit external CSV representation.
The ESM-C aligned NPZ can be passed directly as `--features embeddings.npz`;
its record IDs, vector dimensions, finite available rows and availability mask
are validated. An explicit numeric embedding allowlist is generated automatically
for NPZ, and missing aligned rows retain their failure status. CSV inputs retain
the raw-geometry default.
Classifier failures retain the frozen test denominator. Results report coverage,
per-class support and effectiveness separately from accuracy on predictions.

The Python API also provides hierarchical equal-weight accuracy and a paired
fixed-prediction bootstrap over topology/PDB blocks. Sampling intervals require
multiple independent blocks; repeated random seeds reuse the same proteins.
Full benchmark runs, parameter selection, embedding extraction and external tool
comparisons are separate computational steps.

## Training-only Foldseek reference search

```bash
python -m betlas.benchmarks.structure_search \
  --manifest /data/project/catalog/representatives.csv \
  --exports /data/project/domain-structures/domain_structures.csv \
  --splits /data/project/splits/architecture/splits.csv \
  --foldseek-binary /tools/bin/foldseek --threads 2 --scratch-dir /tmp \
  --out-dir /data/project/runs/foldseek
```

Each split gets its own training-only reference set and held-out queries. The
nearest reference is selected deterministically by bit score, E-value and ID.
No-hit, export failure and backend failure remain visible in the frozen test
denominator. Multichain domains are unavailable until an explicit aggregation
policy is chosen. The command records the binary and search settings and builds
its reference database from the supplied exports. Use native topology labels
through `--label-column` when running that separate task.

Foldseek creates symbolic links in its temporary database tree. `--scratch-dir`
defaults to the system temporary directory; choose a local Linux filesystem with
enough free space for the working databases. Source coordinates and final results
may remain on a separate data disk, including exFAT.

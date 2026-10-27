# ESM-C embeddings

`esmc.py` extracts mean-pooled residue embeddings using the ESM-C 600M model and an explicitly supplied local checkpoint. The model implementation and weights are maintained by the [upstream ESM project](https://github.com/Biohub/esm). This adapter uses the constructor and checkpoint format in [ESM v3.2.1](https://github.com/evolutionaryscale/esm/blob/v3.2.1/esm/pretrained.py).

Install `esm==3.2.1` and its PyTorch dependencies in a separate environment compatible with that upstream release. The Betlas base installation supplies NumPy, pandas, Biopython and tqdm. Obtain the model weights under their upstream terms and pass their local path explicitly. Incompatible checkpoint keys fail loading. The adapter performs inference with the supplied local weights.

```bash
python -m scripts.external_baselines.embeddings.esmc \
  --input-csv inputs/structures.csv \
  --weights /path/to/esmc_600m_checkpoint.pth \
  --out-npz outputs/embeddings/esmc.npz \
  --device cpu --precision fp32
```

Run from the source checkout after installing Betlas. Direct-file invocation with `python scripts/external_baselines/embeddings/esmc.py --help` also works. Model inference requires the optional environment; `--help`, table parsing and alignment work with base dependencies.

The input CSV requires unique, nonempty `record_id` and `structure_path` columns. Structure paths are local mmCIF files, relative to the CSV directory or absolute. Optional `chain_id` selects a preferred author chain. The existing selection rule tries `chain_id`, `selected_chain_id`, `label_chain_id` and `fallback_chain_id` in order, then takes the longest eligible chain. A minimum of 20 residues is required by default; `--min-residues` changes this threshold. Optional `pdb_id` and `filename` metadata default to the input file stem/name.

For example, an independently constructed input table can refer to the bundled synthetic `mini.cif`:

```csv
record_id,structure_path,chain_id
toy_chain,mini.cif,A
```

Use `--min-residues 4` for this tiny synthetic geometry. This illustrates the schema; meaningful model embeddings require appropriate biological inputs.

The NPZ contains `record_id`, `embeddings` (rows aligned to input IDs; 1152 columns), and `esmc_available`. Missing sequences retain zero rows with an availability value of zero. CSV sidecars report selected chains, sequence length, embedding length, truncation, extraction failures and row indices. A JSON sidecar records the input/weight paths and runtime parameters. Use a separate output directory for each job because the sidecar filenames are shared within a directory.

Residue pooling excludes the leading and trailing special tokens. Sequences longer than `--max-length` (default 1022) are truncated and marked in the metadata. `auto` device selection uses CUDA when available; select `cpu` to control execution explicitly. CPU inference uses FP32, and run metadata records both the requested and effective precision. FlashAttention is enabled only for CUDA FP16 when the optional upstream dependency is available. Offline public tests use synthetic inputs to cover parsing, failure status and aligned outputs. Model inference remains outside their scope.

## Exact domain sequences

For CATH domain experiments, supply the exact cropped sequences. The frozen
sequence join exposes `domain_sequence` independently of DSSP:

```bash
python -m scripts.external_baselines.embeddings.esmc \
  --sequence-csv /data/project/sequences.csv \
  --sequence-column domain_sequence --min-residues 1 --weights /models/esmc_600m.pth \
  --out-npz /data/project/embeddings/esmc.npz
```

This mode uses only the supplied sequences. The explicit minimum of one
retains short valid domains; choose any different length exclusion before
evaluating methods. Empty, short and invalid sequences remain in the failure
ledger. The adapter handles sequences independently of geometry status. When
using the feature table,
sequence availability depends on successful DSSP annotation and domain mapping.
For an independent sequence baseline, use the frozen CATH FASTA join provided by
`python -m betlas.benchmarks.sequences --help`. These published sequences may
include residues absent from observed coordinates; report that input-modality
difference. Explicit sequence mode rejects overlength domains by default and
retains them as failures. `--long-sequence-policy truncate` enables truncation.
Structure input uses truncation by default. Inspect `truncated` and
`embedded_length`, and freeze the long-sequence policy before comparisons.
Supply the checkpoint locally for either input mode.

# External beta-barrel detection adapters

These adapters prepare user-supplied inputs, invoke separately installed methods,
and normalize predictions. They retain chain identifiers and residue mappings so
predictions can be traced back to the input structure.

| Method | Input | Adapter |
| --- | --- | --- |
| [Cooper-Beta](cooper_beta/README.md) | Selected chain/model coordinates and fresh DSSP | Native graph decision and separately supplied frozen learned model |
| [Foldseek](foldseek/README.md) | Structures and a reference structure collection or database | Global TMalign search and configurable hit decisions |
| [IsItABarrel](isitabarrel/README.md) | Contact maps or structures | Upstream contact-map scoring and normalized decisions |
| [PRED-TMBB2](pred_tmbb2/README.md) | FASTA or structures | Local JUCHMME single-sequence predictions and topology parsing |

For paired solenoid and barrel studies, use the explicit
[specialist task interfaces](../SPECIALIST_TASKS.md). They preserve unsuccessful
inputs, distinguish barrel geometry from membrane identity, and retain native
hard calls with continuous scores left unavailable. The shared Foldseek
interface uses labelled positive and negative references for either task.

Install Betlas and its Python dependencies before using these source-checkout
scripts. Run commands from the repository root; direct script execution is also
supported. Supply the external executables, source checkouts, model files, and
reference collections before running an adapter.

Structure preparation accepts PDB, ENT, CIF, mmCIF, and gzip-compressed variants.
It uses the first model and protein residues with a CA atom. The default minimum
chain length is 15 residues and can be changed with `--min-residues`. Prepared
inputs contain only residues observed in the supplied structure.

Each method provides a low-level `runner`, a structure preparation command, a
combined structure command, and `evaluate_dataset` for explicit positive and
negative structure directories. Dataset commands write chain-level results,
file-level predictions using the logical OR of chain predictions, and classification
metrics. Betlas geometry compatibility columns use zero or empty placeholders
for measurements unavailable from the external method. Use the method-specific
score columns.
The optional manual-review manifest accepted by Foldseek and PRED-TMBB2 has columns
`filename,final_split,include_for_metrics,policy,reason`, with `final_split` equal
to `positive` or `negative`. All input labels and annotations come from the caller.

Generated CSV metadata contains paths to the inputs supplied by the caller.
Select an appropriate output directory for your own data. Synthetic parser,
command, and structure-conversion checks are in
[`tests/test_external_detection.py`](../../../tests/test_external_detection.py).

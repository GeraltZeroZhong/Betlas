# Local protein sequence search

These adapters run externally installed [NCBI BLASTp](https://www.ncbi.nlm.nih.gov/books/NBK279690/)
and [MMseqs2](https://github.com/soedinglab/MMseqs2) against local
inputs. They retain all reported hits in the upstream tabular format. The source
tree contains adapters and synthetic tests; install each upstream tool separately
under its own licensing and citation terms. Supply your own sequence files and
databases. Search results depend on the installed version, database, and search
parameters.

Run from a source checkout with the repository root on `PYTHONPATH`:

```bash
python -m scripts.external_baselines.sequence_search blastp --help
python -m scripts.external_baselines.sequence_search mmseqs --help
```

After installing NCBI BLAST+ and preparing a local protein database:

```bash
python -m scripts.external_baselines.sequence_search blastp \
  --query queries.faa --database /path/to/protein_database \
  --output outputs/blastp_hits.tsv --threads 4
```

For an MMseqs2 search against a target FASTA or a prepared sequence database:

```bash
python -m scripts.external_baselines.sequence_search mmseqs \
  --query queries.faa --target targets.faa \
  --output outputs/mmseqs_hits.tsv --temporary outputs/mmseqs_tmp --threads 4
```

All searches use supplied local paths, databases and executables. BLASTp uses
E-value `1e-5` and at most 10 targets by default; MMseqs2 uses E-value `1e-3` and
sensitivity `7.5`. Flags expose these settings.
Both use one CPU thread by default. Existing output files are replaced according
to the upstream tool's behavior; use a new output path for each retained search.

## Output schemas

Outputs are headerless TSV files in upstream row order:

- BLASTp: `qseqid, saccver, pident, length, qcovs, evalue, bitscore, sscinames,
  sskingdoms, stitle`. Taxonomy columns depend on the supplied database metadata.
- MMseqs2: `query, target, pident, alnlen, qstart, qend, tstart, tend, evalue, bits`.

`read_hits(path, method="blastp")` or `read_hits(path, method="mmseqs")` loads every
row as a dictionary of strings and rejects malformed column counts. Queries with
no reported hits have no TSV row. Retain the query FASTA for denominator accounting.
An empty TSV is a valid result. Tool failures propagate as unsuccessful commands.

## Synthetic verification

`tests/test_external_sequence_search.py` constructs invented FASTA records and
analytical TSV rows. Temporary fake executables check argument passing and failure
propagation. These tests verify adapter contracts. Assess search accuracy and
biological performance with the installed tools and an appropriate dataset.

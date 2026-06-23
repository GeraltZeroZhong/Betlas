# Smoke Results

Smoke tests used recorded wrapper executions on 2026-05-03 before
these wrappers were integrated under Betlas.

| Method | Input | Result |
| --- | --- | --- |
| BETAWARE | upstream `example/1qj8.fasta` + `example/1qj8.prof` | Pass; predicted TMBB `Yes 1.00` and emitted topology `2-9,23-29,35-46,60-69,78-87,103-114,120-131,135-145`. |
| PRED-TMBB2 HMM | upstream BETAWARE `1qj8.fasta` | Pass; JUCHMME 1.0.6 produced a `VP:` topology line. |
| PRED-TMBB2+HNN | upstream BETAWARE `1qj8.fasta` | Pass; JUCHMME 1.0.6 HNN config completed and produced a `VP:` line. This is a smoke test, not an accuracy check. |
| PROFtmb | Debian package example `example.Q` profile | Pass; tabular output reports length 172, Z-value 10.0, and 8 predicted strands. |
| TMbed | upstream `examples/sample.fasta` + precomputed `examples/sample.h5` | Pass; wrote 15-line `sample.pred` for 5 proteins without downloading ProtT5. |
| PolarBearal3 | RCSB PDB `1A0S.pdb` | Pass; summary reports 18 strands, axis length 25.992, average radius 17.452, plus strand and PyMOL output files. |

Generated files are under ignored `scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/runs/`.

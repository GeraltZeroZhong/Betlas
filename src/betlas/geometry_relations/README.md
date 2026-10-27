# Geometry relationship analysis

These modules compare geometric descriptors and strand-connectivity
representations using supplied structures, tables and saved model outputs.
Install `betlas[geometry]` for Gemmi and NetworkX support.

`packing.PACKING_BLOCKS` defines four measurement groups: global shape, sheet
shape, axis organisation, and continuous circular moments. The packing interface
uses the current descriptor names and preserves unavailable measurements as NaN.
`PackingScaler.fit(training)` learns scales from the supplied training rows;
`transform` applies those fixed scales to subsequent inputs. Distances require
finite active measurements in each included block.

`annotations` builds domain geometry from deposited or DSSP annotations;
`connectivity` and `connection_patterns` describe strand pairing;
`geometry_overlay`, `phenotypes`, `matched`, and `model_overlay` support table
comparisons. `measurement_controls` provides rigid rotations and block-distance
decompositions. `scaffold_alignment` and `validation` accept locally installed
USalign or TM-align executables for structure alignment. `scaffold_cases` renders
coordinate illustrations and requires Matplotlib.

The command-line analyses take a JSON configuration path. Choose the
`source_derived`, `source_runs`, `derived` and `outputs` directories in that
configuration. Run each module with `--help` for its interface.
The measurement APIs also accept geometry or tables in memory, independently of
experiment folders. Graph connectivity and structural alignment describe
distinct properties. Inferring a biological transition requires additional
evidence.

Public tests use synthetic structures and tables. Supply datasets and external
executables separately.

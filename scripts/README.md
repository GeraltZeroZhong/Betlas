# Companion scripts

[external_baselines](external_baselines/README.md) provides adapters for independently installed structure, sequence and membrane-protein methods, plus ESM-C embedding extraction.

Run Python module commands from a source checkout after installing Betlas and the relevant optional dependencies:

```bash
python -m scripts.external_baselines.embeddings.esmc --help
python -m scripts.external_baselines.beta_barrel_detection.foldseek.structure_search --help
```

The source distribution includes these scripts, documentation and shell/C# adapters. The Betlas wheel contains the core `betlas` package; obtain the source checkout or source distribution to use this companion directory. Each method directory documents its upstream implementation, input/output contract and required local executables or model files.

Choose the inputs, reference databases, model weights and output directories for each run. Synthetic tests exercise wrapper behavior using small constructed inputs and local stand-in executables. Running the methods requires the upstream tools and suitable resources.

# Changelog

## 1.1.0

- Unify Python, the default extraction CLI and CATH extraction around one fixed 136-descriptor contract, with circular moments and explicit applicability masks.
- Add learned solenoid recognition, geometric-barrel recognition and barrel-wall strand counting through `betlas predict` and the Python prediction API.
- Add checkpoint installation with pinned versions and integrity checks, local checkpoint selection and offline inference. Strand counting adds 16 radial measurements and a barrel-recognition gate that can be bypassed for known barrels.
- Retire the previous descriptor schema, heuristic fold-rule scoring and topology-diagnostic routes. Regenerate existing tables and fitted models with the current descriptor contract. `grammar list` and `grammar describe` remain available; learned applications use `predict`.
- Include plotting dependencies in the geometry extra and document the complete development installation.

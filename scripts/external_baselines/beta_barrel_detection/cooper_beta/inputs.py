"""Select one exact coordinate model and author chain for specialist runs."""
from __future__ import annotations

from pathlib import Path

from Bio.PDB import MMCIFIO, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from scripts.external_baselines.solenoid.structures import load_structure


def select_structure(source: str | Path, output_dir: str | Path, *, chain_id: str,
                     model_index: int = 0) -> tuple[Path, dict]:
    source = Path(source).resolve()
    structure = load_structure(source)
    models = list(structure.get_models())
    if model_index < 0 or model_index >= len(models):
        raise ValueError("Requested model index is absent")
    model = models[model_index]
    if chain_id not in model:
        raise ValueError("Requested author chain is absent from the selected model")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".cif", ".mmcif"}:
        # Preserve declared polymer positions and other source categories. Only
        # the atom-site rows are restricted; do not regenerate atom-only mmCIF.
        data = MMCIF2Dict(str(source))
        model_numbers = data.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(data["_atom_site.id"]))
        wanted_number = str(model.serial_num)
        rows = [i for i, (chain, number) in enumerate(zip(
            data["_atom_site.auth_asym_id"], model_numbers, strict=True
        )) if chain == chain_id and number == wanted_number]
        if not rows:
            raise ValueError("No atom-site rows match the selected model and author chain")
        for key, values in list(data.items()):
            if key.startswith("_atom_site."):
                data[key] = [values[i] for i in rows]
        selected = output / "selected.cif"
        writer = MMCIFIO()
        writer.set_dict(data)
        writer.save(str(selected))
    else:
        class Selection(Select):
            def accept_model(self, candidate):
                return candidate.id == model.id

            def accept_chain(self, candidate):
                return candidate.id == chain_id

        selected = output / "selected.pdb"
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(selected), Selection(), preserve_atom_numbering=True)
        headers = [line for line in source.read_text().splitlines()
                   if line.startswith(("HEADER", "TITLE ", "CRYST1"))
                   or (line.startswith("SEQRES") and line[11:12] == chain_id)]
        selected.write_text("\n".join(headers) + "\n" + selected.read_text())
    return selected, {"source_path": str(source), "selected_path": str(selected),
                      "author_chain_id": chain_id, "model_index": model_index,
                      "model_serial_number": model.serial_num,
                      "residue_numbering": "preserved", "coordinates": "unmodified"}

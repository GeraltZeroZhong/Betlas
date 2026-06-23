import gzip
import os
import re
import string
import subprocess
import tempfile
import warnings

from Bio import BiopythonWarning
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser, Select
from Bio.PDB.DSSP import DSSP
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa

from ..exceptions import ChainNotFoundError, DsspError, InputValidationError, StructureParseError
from ..runtime import require_dssp_binary

_DSSP_PDB_MMCIF_WARNING_PATTERN = r".*not seem to be an mmCIF file.*"
_DEFAULT_CRYST1 = (
    "CRYST1 1000.000 1000.000 1000.000  90.00  90.00  90.00 P 1           1          \n"
)


def _format_structure_parse_error(file_path: str, error: Exception) -> str:
    detail = str(error)
    if "_atom_site." in detail:
        return (
            f"Failed to parse structure {file_path}: input mmCIF lacks atom-site fields "
            "required by Biopython/DSSP readouts. The packaged mini.cif fixture is for "
            f"grammar/slice smoke tests only. Parser detail: {detail}"
        )
    return f"Failed to parse structure {file_path}: {detail}"


# -------------------------
# Utilities: element / chain
# -------------------------
_TWO_LETTER_ELEMENTS = {
    "CL", "BR", "NA", "MG", "ZN", "FE", "CA", "CU", "NI", "CO", "MN", "SE",
    "SI", "AL", "CD", "HG", "PB", "SR", "CS", "LI", "AG", "AU", "PT", "IR",
    "KR", "XE", "AR", "NE", "HE",
}


def _infer_element_from_atom_name(
    atom_name: str,
    *,
    protein_residue: bool = False,
    residue_name: str = "",
) -> str:
    """
    Infer an element symbol from a PDB atom name.

    Returns Biopython-style capitalization such as ``C``, ``N``, ``O``, ``Cl``,
    or ``Zn``.
    """
    if not atom_name:
        return ""
    s = atom_name.strip()
    if not s:
        return ""

    # Strip a leading digit from names such as "1HG1".
    if s[0].isdigit() and len(s) >= 2:
        s = s[1:]

    # Keep alphabetic characters only.
    s = re.sub(r"[^A-Za-z]", "", s)
    if not s:
        return ""
    s = s.upper()

    if protein_residue:
        # Protein atom names such as CA/CD/HG mean C-alpha/C-delta/H-gamma, not
        # calcium/cadmium/mercury. MSE is the common protein-like exception.
        if residue_name.strip().upper() == "MSE" and s.startswith("SE"):
            return "Se"
        return s[0]

    if len(s) >= 2 and s[:2] in _TWO_LETTER_ELEMENTS:
        return s[0] + s[1].lower()
    return s[0]


def _fill_missing_atom_elements(model) -> int:
    """Fill empty or placeholder ``atom.element`` values and return the count."""
    fixed = 0
    for residue in model.get_residues():
        protein_residue = is_aa(residue, standard=False)
        residue_name = str(residue.get_resname())
        for atom in residue:
            elem = (getattr(atom, "element", "") or "").strip()
            if elem and elem != "X":
                continue
            inf = _infer_element_from_atom_name(
                atom.get_name(),
                protein_residue=protein_residue,
                residue_name=residue_name,
            )
            if inf:
                atom.element = inf
                fixed += 1
    return fixed


def _sanitize_blank_chain_ids(model) -> int:
    """
    Replace blank chain IDs with valid single-character IDs.

    This avoids mkdssp/gemmi failures during non-polymer validation.
    Returns the number of chains updated.
    """
    used = set()
    chains = list(model.get_chains())
    for ch in chains:
        cid = (ch.id or "").strip()
        if cid:
            used.add(cid)

    pool = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    it = (c for c in pool if c not in used)

    fixed = 0
    for ch in chains:
        if (ch.id or "").strip() == "":
            new_id = next(it, "X")
            ch.id = new_id
            used.add(new_id)
            fixed += 1
    return fixed


# -------------------------
# Utilities: PDB header bug workaround
# -------------------------
def _strip_remark_350_to_temp_pdb(in_path: str) -> str:
    """
    Work around a Biopython ``parse_pdb_header`` bug triggered by some PDBs.

    Even with ``get_header=False``, some versions or code paths still hit the
    ``currentBiomolecule`` bug. As a fallback, drop ``REMARK 350`` records and
    parse the structure again.

    Returns the temporary file path; the caller is responsible for deleting it.
    """
    fd, out_path = tempfile.mkstemp(suffix=".pdb")
    with open(in_path, errors="ignore") as fin, os.fdopen(fd, "w") as fout:
        for line in fin:
            if line.startswith("REMARK 350"):
                continue
            fout.write(line)
    return out_path


def _decompress_gzip_to_temp_if_needed(in_path: str) -> str | None:
    with open(in_path, "rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            return None

    suffixes = [suffix.lower() for suffix in os.path.basename(in_path).split(".")[1:]]
    if suffixes and suffixes[-1] == "gz" and len(suffixes) >= 2:
        suffix = f".{suffixes[-2]}"
    else:
        suffix = os.path.splitext(in_path)[1] or ".pdb"
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    try:
        with gzip.open(in_path, "rb") as source, os.fdopen(fd, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    except Exception:
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
    return out_path


def _mmcif_value_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_mmcif_ins_code(value: str) -> str:
    return " " if value in {"", ".", "?"} else value


def _atom_site_label_to_auth_residue_map(
    cif_path: str,
) -> dict[tuple[str, str], tuple[str, tuple[str, int, str]]]:
    mmcif = MMCIF2Dict(cif_path)
    label_asym_ids = _mmcif_value_list(mmcif.get("_atom_site.label_asym_id"))
    label_seq_ids = _mmcif_value_list(mmcif.get("_atom_site.label_seq_id"))
    auth_asym_ids = _mmcif_value_list(mmcif.get("_atom_site.auth_asym_id"))
    auth_seq_ids = _mmcif_value_list(mmcif.get("_atom_site.auth_seq_id"))
    insertion_codes = _mmcif_value_list(mmcif.get("_atom_site.pdbx_PDB_ins_code"))
    if len(insertion_codes) != len(label_asym_ids):
        insertion_codes = ["?"] * len(label_asym_ids)

    mapping: dict[tuple[str, str], tuple[str, tuple[str, int, str]]] = {}
    for label_asym, label_seq, auth_asym, auth_seq, insertion_code in zip(
        label_asym_ids,
        label_seq_ids,
        auth_asym_ids,
        auth_seq_ids,
        insertion_codes,
        strict=False,
    ):
        if label_seq in {"", ".", "?"} or auth_seq in {"", ".", "?"}:
            continue
        try:
            residue_number = int(auth_seq)
        except ValueError:
            continue
        mapping.setdefault(
            (label_asym, label_seq),
            (auth_asym, (" ", residue_number, _normalize_mmcif_ins_code(insertion_code))),
        )
    return mapping


def _dssp_mmcif_summary_to_secondary_structure(
    mmcif: dict[str, object],
    label_to_auth_residue: dict[tuple[str, str], tuple[str, tuple[str, int, str]]],
) -> dict[tuple[str, tuple[str, int, str]], str]:
    label_asym_ids = _mmcif_value_list(mmcif.get("_dssp_struct_summary.label_asym_id"))
    label_seq_ids = _mmcif_value_list(mmcif.get("_dssp_struct_summary.label_seq_id"))
    secondary_structures = _mmcif_value_list(
        mmcif.get("_dssp_struct_summary.secondary_structure")
    )

    secondary_structure: dict[tuple[str, tuple[str, int, str]], str] = {}
    for label_asym, label_seq, ss_code in zip(
        label_asym_ids,
        label_seq_ids,
        secondary_structures,
        strict=False,
    ):
        mapped = label_to_auth_residue.get((label_asym, label_seq))
        if mapped is None:
            continue
        auth_asym, residue_id = mapped
        secondary_structure[(auth_asym, residue_id)] = "-" if ss_code in {".", "?"} else ss_code
    return secondary_structure


# -------------------------
# DSSP: export protein only
# -------------------------
class _ProteinOnlySelect(Select):
    """Export only amino-acid residues, including non-standard residues like MSE."""
    def accept_residue(self, residue):
        return 1 if is_aa(residue, standard=False) else 0

    def accept_atom(self, atom):
        return 1


class ProteinLoader:
    """
    Load PDB/mmCIF structures, run DSSP, and extract per-chain CA data.
    """

    def __init__(
        self,
        file_path,
        model_id=0,
        dssp_bin=None,
        fail_on_dssp_error=True,
        strict_chain: bool = True,
    ):
        self.file_path = file_path
        self.model_id = model_id
        self.dssp_bin = dssp_bin
        self.fail_on_dssp_error = bool(fail_on_dssp_error)
        self.strict_chain = bool(strict_chain)

        self.structure = None
        self.model = None
        self.secondary_structure = None
        self.secondary_structure_error = None

        self._load_structure()

    def _load_structure(self):
        if not os.path.exists(self.file_path):
            raise InputValidationError(f"Structure file not found: {self.file_path}")

        try:
            input_tmp = _decompress_gzip_to_temp_if_needed(self.file_path)
        except (OSError, EOFError) as exc:
            raise StructureParseError(f"Failed to decompress structure {self.file_path}: {exc}") from None
        parse_path = input_tmp or self.file_path
        ext = os.path.splitext(parse_path)[1].lower()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BiopythonWarning)
                if ext in [".cif", ".mmcif"]:
                    parser = MMCIFParser(QUIET=True)
                    self.structure = parser.get_structure("struct", parse_path)
                else:
                    # Important: disable header parsing.
                    parser = PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
                    self.structure = parser.get_structure("struct", parse_path)

            self.model = self.structure[self.model_id]
            return

        except Exception as e:
            # Fallback: only PDB files go through the REMARK 350 stripping path.
            if ext not in [".cif", ".mmcif"]:
                tmp = None
                try:
                    tmp = _strip_remark_350_to_temp_pdb(parse_path)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", BiopythonWarning)
                        parser = PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
                        self.structure = parser.get_structure("struct", tmp)
                    self.model = self.structure[self.model_id]
                    return
                except Exception as e2:
                    raise StructureParseError(
                        f"Failed to parse structure {self.file_path}: {e2}"
                    ) from None
                finally:
                    if tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass

            raise StructureParseError(_format_structure_parse_error(self.file_path, e)) from None
        finally:
            if input_tmp and os.path.exists(input_tmp):
                try:
                    os.remove(input_tmp)
                except OSError:
                    pass

    def _has_multichar_chain_ids(self) -> bool:
        return any(len(str(chain.id)) > 1 for chain in self.model.get_chains())

    def _export_protein_only_structure(self) -> str:
        _sanitize_blank_chain_ids(self.model)
        _fill_missing_atom_elements(self.model)

        if self._has_multichar_chain_ids():
            fd, tmp_path = tempfile.mkstemp(suffix=".cif")
            os.close(fd)
            io = MMCIFIO()
            io.set_structure(self.model)
            io.save(tmp_path, select=_ProteinOnlySelect())
            return tmp_path

        fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        with os.fdopen(fd, "w") as handle:
            handle.write("HEADER    GENERATED BY LOADER                         \n")
            handle.write(_DEFAULT_CRYST1)
            io = PDBIO()
            io.set_structure(self.model)
            io.save(handle, select=_ProteinOnlySelect())
        return tmp_path

    def _run_dssp(self, tmp_path: str) -> dict[tuple[str, tuple[str, int, str]], str]:
        dssp_bin = require_dssp_binary(self.dssp_bin)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_DSSP_PDB_MMCIF_WARNING_PATTERN,
                    category=UserWarning,
                )
                dssp_result = DSSP(self.model, tmp_path, dssp=dssp_bin)
        except Exception:
            if tmp_path.lower().endswith((".cif", ".mmcif")):
                return self._run_dssp_mmcif_output(tmp_path, dssp_bin)
            raise
        return {dssp_key: str(dssp_result[dssp_key][2]) for dssp_key in dssp_result.keys()}

    def _run_dssp_mmcif_output(
        self,
        tmp_path: str,
        dssp_bin: str,
    ) -> dict[tuple[str, tuple[str, int, str]], str]:
        label_to_auth_residue = _atom_site_label_to_auth_residue_map(tmp_path)
        result = subprocess.run(
            [dssp_bin, tmp_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stderr.strip():
            warnings.warn(result.stderr.strip(), UserWarning, stacklevel=2)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"{dssp_bin} exited {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("DSSP failed to produce an mmCIF output")

        fd, dssp_mmcif_path = tempfile.mkstemp(suffix=".dssp.cif")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(result.stdout)
            mmcif = MMCIF2Dict(dssp_mmcif_path)
        finally:
            if os.path.exists(dssp_mmcif_path):
                try:
                    os.remove(dssp_mmcif_path)
                except OSError:
                    pass

        secondary_structure = _dssp_mmcif_summary_to_secondary_structure(
            mmcif,
            label_to_auth_residue,
        )
        if not secondary_structure:
            raise RuntimeError("DSSP mmCIF output did not contain residue summaries")
        return secondary_structure

    def _run_secondary_structure(self):
        if self.secondary_structure is not None:
            return

        tmp_path = None
        try:
            # Export protein ATOM records only. Dropping HETATM helps avoid
            # nonpoly_scheme strand/duplicate key issues.
            tmp_path = self._export_protein_only_structure()
            self.secondary_structure = self._run_dssp(tmp_path)

        except Exception as e:
            self.secondary_structure_error = f"DSSP failed for {os.path.basename(self.file_path)}: {e}"
            if self.fail_on_dssp_error:
                raise DsspError(self.secondary_structure_error) from e
            warnings.warn(self.secondary_structure_error, RuntimeWarning, stacklevel=2)
            self.secondary_structure = {}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def available_chains(self) -> list[str]:
        """Return chain IDs available in the selected model."""
        return [str(chain.id) for chain in self.model.get_chains()]

    def get_ca_data(self, chain_id, *, strict_chain: bool | None = None):
        chain = self.model[chain_id] if chain_id in self.model else None
        if chain is None:
            chains = list(self.model.get_chains())
            strict = self.strict_chain if strict_chain is None else bool(strict_chain)
            if not strict and len(chains) == 1:
                chain = chains[0]
            else:
                available = ", ".join(self.available_chains()) or "none"
                raise ChainNotFoundError(
                    f"Chain {chain_id!r} not found in {os.path.basename(self.file_path)}. "
                    f"Available chains: {available}."
                )

        if self.secondary_structure is None:
            self._run_secondary_structure()

        data = []
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            if "CA" not in res:
                continue

            dssp_key = (chain.id, res.id)
            ss_code = "-"
            if self.secondary_structure and dssp_key in self.secondary_structure:
                ss_code = self.secondary_structure[dssp_key]

            data.append(
                {
                    "res_id": res.id[1],
                    "hetflag": str(res.id[0]).strip(),
                    "insertion_code": "" if str(res.id[2]).strip() == "" else str(res.id[2]),
                    "res_uid": f"{res.id[0]}:{res.id[1]}:{res.id[2]}",
                    "chain": chain.id,
                    "coord": res["CA"].get_coord().astype(float).tolist(),
                    "is_sheet": ss_code in ("E", "B"),
                }
            )
        return data

    def get_chain_data(self, chain_id, *, strict_chain: bool | None = None):
        return self.get_ca_data(chain_id, strict_chain=strict_chain)

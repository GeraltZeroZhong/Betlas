from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

_COMPAT_BENCHMARK_FLAG = "allowed_for_" + "pub" + "lication_benchmark"


@dataclass(frozen=True)
class ResidueRecord:
    pdb_id: str
    chain_id: str
    auth_seq_id: int
    label_seq_id: int | None
    insertion_code: str
    residue_name: str
    coord_ca: tuple[float, float, float]

    @property
    def residue_uid(self) -> str:
        ins = self.insertion_code or ""
        return f"{self.chain_id}:{self.auth_seq_id}:{ins}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SecondaryStructureElement:
    element_id: str
    element_type: str
    chain_id: str
    start_auth_seq_id: int
    end_auth_seq_id: int
    residue_indices: tuple[int, ...]
    sheet_id: str = ""
    sheet_range_id: str = ""
    sense_to_previous: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SheetPatch:
    sheet_id: str
    strand_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DomainCandidate:
    record_id: str
    pdb_id: str
    chain_id: str
    domain_id: str
    residue_ranges: str
    fold_label_final: str
    evidence_level: str
    label_source_primary: str
    label_source_supporting: str = ""
    cath_code: str = ""
    cath_architecture_code: str = ""
    cath_topology_code: str = ""
    cath_homology_code: str = ""
    cath_s35_cluster_id: str = ""
    cath_s35_source: str = ""
    cath_status: str = ""
    cath_name: str = ""
    assembly_id: str = "1"
    model_id: int = 0
    qc_status: str = "external_source_unreviewed"
    allowed_for_benchmark: bool = False
    discovered_by_betlas: bool = False
    label_conflict_notes: str = ""

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> DomainCandidate:
        return cls(
            record_id=str(row.get("record_id", "")),
            pdb_id=str(row.get("pdb_id", "")).lower(),
            chain_id=str(row.get("chain_id", "")),
            domain_id=str(row.get("domain_id", "")),
            residue_ranges=str(row.get("residue_ranges", "")),
            fold_label_final=str(row.get("fold_label_final", "")),
            evidence_level=str(row.get("evidence_level", "")),
            label_source_primary=str(row.get("label_source_primary", "")),
            label_source_supporting=str(row.get("label_source_supporting", "")),
            cath_code=str(row.get("cath_code", "")),
            cath_architecture_code=str(row.get("cath_architecture_code", "")),
            cath_topology_code=str(row.get("cath_topology_code", "")),
            cath_homology_code=str(row.get("cath_homology_code", "")),
            cath_s35_cluster_id=str(row.get("cath_s35_cluster_id", "")),
            cath_s35_source=str(row.get("cath_s35_source", "")),
            cath_status=str(row.get("cath_status", "")),
            cath_name=str(row.get("cath_name", "")),
            assembly_id=str(row.get("assembly_id", "1") or "1"),
            model_id=int(row.get("model_id", 0) or 0),
            qc_status=str(row.get("qc_status", "external_source_unreviewed")),
            allowed_for_benchmark=str(row.get("allowed_for_benchmark", row.get(_COMPAT_BENCHMARK_FLAG, "False"))).lower()
            in {"1", "true", "yes"},
            discovered_by_betlas=str(row.get("discovered_by_betlas", "False")).lower()
            in {"1", "true", "yes"},
            label_conflict_notes=str(row.get("label_conflict_notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AxisHypothesis:
    name: str
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    support_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructureGeometry:
    domain: DomainCandidate
    residues: tuple[ResidueRecord, ...]
    beta_segments: tuple[SecondaryStructureElement, ...]
    helices: tuple[SecondaryStructureElement, ...]
    sheet_patches: tuple[SheetPatch, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.to_dict(),
            "residues": [residue.to_dict() for residue in self.residues],
            "beta_segments": [segment.to_dict() for segment in self.beta_segments],
            "helices": [helix.to_dict() for helix in self.helices],
            "sheet_patches": [sheet.to_dict() for sheet in self.sheet_patches],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GeometrySignature:
    domain: DomainCandidate
    features: dict[str, float | int | str]
    fold_scores: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        row = self.domain.to_dict()
        row.update(self.features)
        for label, score in self.fold_scores.items():
            row[f"betlas_rule_score_{label}"] = float(score)
        row["betlas_warnings"] = ";".join(self.warnings)
        return row

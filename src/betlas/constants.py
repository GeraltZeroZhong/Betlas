from __future__ import annotations

from pathlib import Path

from .schema import BETLAS_PREFIX

FOLD_LABELS: tuple[str, ...] = (
    "beta_barrel",
    "beta_prism",
    "beta_propeller",
    "jelly_roll",
    "beta_solenoid",
    "beta_sandwich",
    "tim_like_beta_alpha_barrel",
)

DIAGNOSTIC_PREFIX = BETLAS_PREFIX

DEFAULT_CATH_DIR = Path("inputs/cath")
DEFAULT_MMCIF_DIR = Path("inputs/mmcif")
DEFAULT_LABELS_CSV = Path("outputs/labels.csv")
DEFAULT_FEATURES_CSV = Path("outputs/features.csv")
DEFAULT_RUN_DIR = Path("outputs/benchmark")

CATH_URLS = {
    "all": "https://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz",
    "names": "https://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-names.gz",
    "s35": "https://download.cathdb.info/cath/releases/daily-release/newest/cath-b-s35-newest.gz",
}

STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
    "SEC",
    "PYL",
}

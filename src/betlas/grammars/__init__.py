"""Inspectable families of the current geometric descriptors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..features.current import FEATURE_FAMILIES, FAMILY_DEFINITIONS
from ..features.extract import extract_classifier_features
from ..models import StructureGeometry


@dataclass(frozen=True)
class GrammarSpec:
    name: str
    summary: str
    columns: tuple[str, ...]
    math_summary: str
    inputs: tuple[str, ...] = ("C-alpha coordinates and secondary-structure annotations",)
    outputs: tuple[str, ...] = ("Measured descriptors with explicit applicability",)
    column_prefixes: tuple[str, ...] = ()
    implementation_notes: tuple[str, ...] = ("Missing geometric measurements remain NaN.",)

    def matches_column(self, column: str) -> bool:
        return column in self.columns

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolved_columns"] = list(self.columns)
        return result


_GRAMMARS = tuple(GrammarSpec(name, FAMILY_DEFINITIONS[name], columns, FAMILY_DEFINITIONS[name])
                  for name, columns in FEATURE_FAMILIES.items())


def list_grammars() -> tuple[GrammarSpec, ...]:
    return _GRAMMARS


def get_grammar(name: str) -> GrammarSpec:
    for grammar in _GRAMMARS:
        if grammar.name == name:
            return grammar
    raise ValueError(f"Unknown Betlas descriptor family {name!r}; expected one of {list(FEATURE_FAMILIES)}")


def compute_grammar_features(geometry: StructureGeometry, grammars=None) -> dict[str, Any]:
    features = extract_classifier_features(geometry)
    if grammars is None:
        return features
    names = (grammars,) if isinstance(grammars, str) else tuple(grammars)
    columns = {column for name in names for column in get_grammar(name).columns}
    return {name: value for name, value in features.items() if name in columns}


__all__ = ["GrammarSpec", "compute_grammar_features", "get_grammar", "list_grammars"]

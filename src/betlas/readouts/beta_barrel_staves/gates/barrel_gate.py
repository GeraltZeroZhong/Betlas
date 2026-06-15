from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig


@dataclass(frozen=True)
class BarrelGateDecision:
    enabled: bool
    passed: bool
    result: str
    reason: str
    stage: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "passed": self.passed,
            "result": self.result,
            "reason": self.reason,
            "stage": self.stage,
            "score": self.score,
        }


def disabled_gate_decision() -> BarrelGateDecision:
    return BarrelGateDecision(
        enabled=False,
        passed=True,
        result="SKIPPED",
        reason="External barrel gate is disabled.",
    )


def error_gate_decision(reason: str) -> BarrelGateDecision:
    return BarrelGateDecision(
        enabled=True,
        passed=False,
        result="ERROR",
        reason=reason,
        stage="barrel_gate",
    )


def bypassed_error_gate_decision(reason: str) -> BarrelGateDecision:
    return BarrelGateDecision(
        enabled=True,
        passed=True,
        result="ERROR_BYPASSED",
        reason=reason,
        stage="barrel_gate",
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_value(row: object, key: str, default: object = "") -> object:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _rows_by_chain(rows: Iterable[object]) -> dict[str, object]:
    by_chain: dict[str, object] = {}
    for row in rows:
        chain = str(_row_value(row, "chain", ""))
        if chain:
            by_chain[chain] = row
    return by_chain


def run_external_barrel_gate(file_path: str, cfg: AppConfig) -> dict[str, BarrelGateDecision]:
    """
    Run an optional external barrel/non-barrel prefilter.

    Betlas is self-contained and does not bundle another Betlas project as a
    dependency. The hook is kept so cached payloads and future external adapters
    have a stable place in the readout contract.
    """
    if not cfg.barrel_gate.enabled:
        return {}

    path = Path(file_path).name
    raise RuntimeError(
        "External barrel gate is enabled, but Betlas does not bundle an external "
        f"gate adapter for {path}. Set `barrel_gate.enabled=false` to run the "
        "self-contained beta-barrel-staves readout."
    )


def gate_decision_for_chain(
    decisions: Mapping[str, BarrelGateDecision],
    chain_id: str,
    *,
    fallback_chain_id: str | None = None,
) -> BarrelGateDecision:
    decision = decisions.get(chain_id)
    if decision is not None:
        return decision
    if fallback_chain_id:
        decision = decisions.get(fallback_chain_id)
        if decision is not None:
            return decision
    return error_gate_decision(f"External barrel gate produced no row for chain {chain_id!r}.")


def gate_from_payload(payload: Mapping[str, Any]) -> BarrelGateDecision | None:
    raw_gate = payload.get("barrel_gate")
    if raw_gate is None:
        return None
    if not isinstance(raw_gate, Mapping):
        return error_gate_decision("Invalid barrel gate payload.")
    return BarrelGateDecision(
        enabled=bool(raw_gate.get("enabled", True)),
        passed=bool(raw_gate.get("passed", False)),
        result=str(raw_gate.get("result", "")),
        reason=str(raw_gate.get("reason", "")),
        stage=str(raw_gate.get("stage", "")),
        score=_safe_float(raw_gate.get("score", 0.0)),
    )

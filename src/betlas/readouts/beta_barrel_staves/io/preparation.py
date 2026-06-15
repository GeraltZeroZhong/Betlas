from __future__ import annotations

import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from ..config import AppConfig
from ..gates.barrel_gate import (
    bypassed_error_gate_decision,
    disabled_gate_decision,
    gate_decision_for_chain,
    run_external_barrel_gate,
)
from ..runtime import require_dssp_binary
from .loader import ProteinLoader
from .prepare_cache import load_prepare_payloads, store_prepare_payloads

DEFAULT_PREPARE_BATCH_SIZE = 16


@dataclass(frozen=True)
class PrepareFailure:
    message: str


def prepare_one_file(file_path: str, cfg: AppConfig) -> list[dict[str, object]] | PrepareFailure:
    """Parse a structure once, run DSSP once, and produce per-chain payloads."""
    filename = os.path.basename(file_path)
    try:
        cfg = deepcopy(cfg)
        cfg.runtime.dssp_bin_path = require_dssp_binary(cfg.runtime.dssp_bin_path)
        cached_payloads = load_prepare_payloads(file_path, cfg)
        if cached_payloads is not None:
            return cached_payloads

        loader = ProteinLoader(
            file_path,
            dssp_bin=cfg.runtime.dssp_bin_path,
            fail_on_dssp_error=cfg.runtime.fail_on_dssp_error,
        )
    except Exception as exc:
        return PrepareFailure(f"{filename}: {exc}")

    barrel_gate_decisions = {}
    gate_bypassed_after_error = False
    if cfg.barrel_gate.enabled:
        try:
            barrel_gate_decisions = run_external_barrel_gate(file_path, cfg)
        except Exception as exc:
            if cfg.barrel_gate.fail_on_error:
                return PrepareFailure(f"{filename}: {exc}")
            barrel_gate_decisions = {}
            gate_bypassed_after_error = True

    payloads: list[dict[str, object]] = []
    for chain_index, chain in enumerate(loader.model):
        chain_id = chain.id
        try:
            residues_data = loader.get_chain_data(chain_id)
        except Exception as exc:
            return PrepareFailure(f"{filename}: {exc}")

        effective_chain_id = (
            str(residues_data[0].get("chain", chain_id)) if residues_data else str(chain.id)
        )

        if cfg.barrel_gate.enabled:
            gate_decision = gate_decision_for_chain(
                barrel_gate_decisions,
                str(chain_id),
                fallback_chain_id=effective_chain_id,
            )
        else:
            gate_decision = disabled_gate_decision()
        if gate_bypassed_after_error:
            gate_decision = bypassed_error_gate_decision(
                "External barrel gate failed, but `barrel_gate.fail_on_error=false` allowed analysis to continue."
            )

        payloads.append(
            {
                "filename": filename,
                "source_path": file_path,
                "chain": effective_chain_id,
                "_chain_index": chain_index,
                "dssp_status": "error" if loader.secondary_structure_error else "ok",
                "dssp_error": loader.secondary_structure_error or "",
                "residues_data": residues_data,
                "barrel_gate": gate_decision.to_dict(),
            }
        )

    has_dssp_error = any(str(payload.get("dssp_error", "")) for payload in payloads)
    if not gate_bypassed_after_error and not has_dssp_error:
        store_prepare_payloads(file_path, cfg, payloads)
    return payloads


def resolve_prepare_batch_size(cfg: AppConfig) -> int:
    batch_size = getattr(cfg.runtime, "prepare_batch_size", DEFAULT_PREPARE_BATCH_SIZE)
    try:
        return max(1, int(batch_size))
    except (TypeError, ValueError):
        return DEFAULT_PREPARE_BATCH_SIZE


def iter_file_batches(file_list: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(file_list), batch_size):
        yield file_list[index : index + batch_size]


def prepare_file_batch(file_paths: list[str], cfg: AppConfig) -> dict[str, object]:
    """Prepare a batch of files in one worker call."""
    payloads: list[dict[str, object]] = []
    errors: list[str] = []

    for file_path in file_paths:
        result = prepare_one_file(file_path, cfg)
        if isinstance(result, PrepareFailure):
            errors.append(result.message)
            continue
        payloads.extend(result)

    return {
        "payloads": payloads,
        "errors": errors,
        "processed": len(file_paths),
    }

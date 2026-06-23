from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadoutSpec:
    name: str
    summary: str
    main: Callable[[list[str] | None], None]
    input_protocol: str
    output_protocol: str
    requires: str = ""
    surface: str = "native"


def _beta_barrel_staves_main(argv: list[str] | None = None) -> None:
    from .beta_barrel_staves.cli import main

    main(argv)


def _beta_barrel_detection_main(argv: list[str] | None = None) -> None:
    from .beta_barrel_detection.cli import main

    main(argv)


def _topology_diagnostics_main(
    argv: list[str] | None = None,
    *,
    prog: str = "betlas readout topology-diagnostics",
    fixed_mode: str | None = None,
) -> None:
    from .topology_diagnostics.cli import main

    main(argv, prog=prog, fixed_mode=fixed_mode)


def _mode_main(mode: str) -> Callable[[list[str] | None], None]:
    def run(argv: list[str] | None = None) -> None:
        args = list(argv or [])
        alias = {
            "ambiguity": "topology-ambiguity",
            "continuous": "fold-continuous-scores",
            "mixed": "mixed-topology",
        }.get(mode, "topology-diagnostics")
        if any(arg == "--mode" or arg.startswith("--mode=") for arg in args):
            raise ValueError(
                f"{alias} has a fixed topology mode; use topology-diagnostics --mode ... "
                "to choose a different topology diagnostics subset"
            )
        _topology_diagnostics_main(args, prog=f"betlas readout {alias}", fixed_mode=mode)

    return run


READOUTS: dict[str, ReadoutSpec] = {
    "beta-barrel-detection": ReadoutSpec(
        name="beta-barrel-detection",
        summary="Betlas native beta-barrel chain detection readout.",
        main=_beta_barrel_detection_main,
        input_protocol="PDB/mmCIF structure file or directory",
        output_protocol="CSV rows with beta-barrel-like geometry decision, heuristic score, stage, gate, and layer evidence",
        requires="DSSP/mkdssp",
    ),
    "beta-barrel-staves": ReadoutSpec(
        name="beta-barrel-staves",
        summary="Secondary readout for beta-barrel strand/stave count.",
        main=_beta_barrel_staves_main,
        input_protocol="PDB/mmCIF structure file or directory",
        output_protocol="CSV rows with candidate stave count, heuristic confidence, gate status, and slice/layer evidence",
        requires="DSSP/mkdssp",
    ),
    "fold-continuous-scores": ReadoutSpec(
        name="fold-continuous-scores",
        summary="Continuous jelly-rollness, sandwichness, and barrel-likeness scores.",
        main=_mode_main("continuous"),
        input_protocol="Betlas feature CSV",
        output_protocol="CSV columns for continuous topology scores and basis JSON",
    ),
    "mixed-topology": ReadoutSpec(
        name="mixed-topology",
        summary="Hybrid-domain and mixed-local-topology detection.",
        main=_mode_main("mixed"),
        input_protocol="Betlas feature CSV",
        output_protocol="CSV columns for mixed-topology score, flag, types, and audit priority",
    ),
    "topology-ambiguity": ReadoutSpec(
        name="topology-ambiguity",
        summary="Boundary-region ambiguity score from probabilities, grammar conflicts, and neighbors.",
        main=_mode_main("ambiguity"),
        input_protocol="Betlas feature CSV with optional prediction CSV",
        output_protocol="CSV columns for ambiguity score, probability summaries, rule conflict, and neighbor evidence",
    ),
    "topology-diagnostics": ReadoutSpec(
        name="topology-diagnostics",
        summary="All topology diagnostic secondary readouts in one table.",
        main=_topology_diagnostics_main,
        input_protocol="Betlas feature CSV with optional prediction CSV",
        output_protocol="CSV table combining ambiguity, continuous topology, and mixed-topology diagnostics",
    ),
}


def list_readouts() -> list[ReadoutSpec]:
    return [READOUTS[name] for name in sorted(READOUTS)]


def get_readout(name: str) -> ReadoutSpec:
    try:
        return READOUTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(READOUTS))
        raise ValueError(f"unknown readout {name!r}; available readouts: {available}") from exc


def run_readout(name: str, argv: list[str] | None = None) -> None:
    get_readout(name).main(argv)

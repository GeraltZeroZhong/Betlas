from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadoutSpec:
    name: str
    summary: str
    main: Callable[[list[str] | None], None]


def _beta_barrel_staves_main(argv: list[str] | None = None) -> None:
    from .beta_barrel_staves.cli import main

    main(argv)


def _beta_barrel_detection_main(argv: list[str] | None = None) -> None:
    from .beta_barrel_detection.cli import main

    main(argv)


def _topology_diagnostics_main(argv: list[str] | None = None) -> None:
    from .topology_diagnostics.cli import main

    main(argv)


def _bfvd_scan_main(argv: list[str] | None = None) -> None:
    from .bfvd_scan.cli import main

    main(argv)


def _mode_main(mode: str) -> Callable[[list[str] | None], None]:
    def run(argv: list[str] | None = None) -> None:
        args = list(argv or [])
        if not any(arg == "--mode" or arg.startswith("--mode=") for arg in args):
            args = ["--mode", mode, *args]
        _topology_diagnostics_main(args)

    return run


READOUTS: dict[str, ReadoutSpec] = {
    "beta-barrel-detection": ReadoutSpec(
        name="beta-barrel-detection",
        summary="Betlas native beta-barrel chain detection readout.",
        main=_beta_barrel_detection_main,
    ),
    "beta-barrel-staves": ReadoutSpec(
        name="beta-barrel-staves",
        summary="Secondary readout for beta-barrel strand/stave count.",
        main=_beta_barrel_staves_main,
    ),
    "bfvd-viral-beta-fold-scan": ReadoutSpec(
        name="bfvd-viral-beta-fold-scan",
        summary="Prospective BFVD viral beta-fold grammar scan and topology audit.",
        main=_bfvd_scan_main,
    ),
    "fold-continuous-scores": ReadoutSpec(
        name="fold-continuous-scores",
        summary="Continuous jelly-rollness, sandwichness, and barrel-likeness scores.",
        main=_mode_main("continuous"),
    ),
    "mixed-topology": ReadoutSpec(
        name="mixed-topology",
        summary="Hybrid-domain and mixed-local-topology detection.",
        main=_mode_main("mixed"),
    ),
    "topology-ambiguity": ReadoutSpec(
        name="topology-ambiguity",
        summary="Boundary-region ambiguity score from probabilities, grammar conflicts, and neighbors.",
        main=_mode_main("ambiguity"),
    ),
    "topology-diagnostics": ReadoutSpec(
        name="topology-diagnostics",
        summary="All topology diagnostic secondary readouts in one table.",
        main=_topology_diagnostics_main,
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

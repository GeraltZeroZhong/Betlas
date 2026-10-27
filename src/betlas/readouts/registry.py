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


READOUTS: dict[str, ReadoutSpec] = {
    "beta-barrel-detection": ReadoutSpec(
        name="beta-barrel-detection",
        summary="Betlas native beta-barrel chain detection readout.",
        main=_beta_barrel_detection_main,
        input_protocol="PDB/mmCIF structure file or directory",
        output_protocol="CSV rows with beta-barrel-like geometry decision, uncalibrated heuristic score, stage, gate, and layer evidence",
        requires="DSSP/mkdssp",
    ),
    "beta-barrel-staves": ReadoutSpec(
        name="beta-barrel-staves",
        summary="Secondary readout for beta-barrel strand/stave count.",
        main=_beta_barrel_staves_main,
        input_protocol="PDB/mmCIF structure file or directory",
        output_protocol="CSV rows with candidate stave count, uncalibrated heuristic confidence, gate status, and slice/layer evidence",
        requires="DSSP/mkdssp",
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

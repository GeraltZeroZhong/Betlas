from __future__ import annotations

import pytest

from betlas import cli
from betlas.grammars import list_grammars


def test_public_cli_commands_provide_help(capsys) -> None:
    help_commands = [
        ["--help"],
        ["grammar", "--help"],
        ["slice", "--help"],
        ["assets", "--help"],
        ["assets", "download", "--help"],
        ["assets", "check-esmc", "--help"],
        ["readout", "beta-barrel-detection", "--help"],
        ["readout", "beta-barrel-staves", "--help"],
        ["predict", "solenoid", "--help"],
        ["predict", "barrel", "--help"],
        ["predict", "barrel-strands", "--help"],
        ["checkpoints", "install", "--help"],
    ]
    for command in help_commands:
        with pytest.raises(SystemExit) as exc:
            cli.main(command)
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out.lower()


def test_public_cli_exposes_root_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("betlas ")






def test_public_cli_user_errors_do_not_print_tracebacks(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["grammar", "describe", "does-not-exist"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err

    with pytest.raises(SystemExit) as exc:
        cli.main(["readout", "does-not-exist"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_grammar_descriptions_are_complete_and_public(capsys) -> None:
    for spec in list_grammars():
        assert spec.math_summary
        assert spec.inputs
        assert spec.outputs
        cli.main(["grammar", "describe", spec.name])
        assert spec.name in capsys.readouterr().out


def test_retired_grammar_score_is_not_exposed():
    with pytest.raises(SystemExit) as exc:
        cli.main(["grammar", "score"])
    assert exc.value.code == 2


def test_prediction_requires_chain_before_loading_checkpoint(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict", "solenoid", "--structure", "missing.cif", "--offline"])
    assert exc.value.code == 2
    assert "--chain is required" in capsys.readouterr().err

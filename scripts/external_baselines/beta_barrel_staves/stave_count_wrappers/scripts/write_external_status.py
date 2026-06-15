#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
SRC_DIR = REPO / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betlas.readouts.beta_barrel_staves.external_baselines import (  # noqa: E402
    build_external_baseline_manifest,
    build_external_method_manifest,
    write_json,
)


def parse_label_path(values: list[str] | None) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label:
            raise ValueError(f"Missing label in {value!r}")
        parsed[label] = Path(path).expanduser()
    return parsed


def parse_command(value: str | None) -> Any:
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_method_manifests(paths: list[Path] | None) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for path in paths or []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        methods[str(payload.get("method") or path.stem)] = payload
    return methods


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write external baseline status JSON.")
    sub = parser.add_subparsers(dest="mode", required=True)

    method = sub.add_parser("method", help="Write one method status manifest.")
    method.add_argument("--out", type=Path, required=True)
    method.add_argument("--method", required=True)
    method.add_argument("--status", required=True)
    method.add_argument("--return-code", type=int)
    method.add_argument("--runtime-seconds", type=float)
    method.add_argument("--command-json")
    method.add_argument("--input", action="append", default=[])
    method.add_argument("--output", action="append", default=[])
    method.add_argument("--message", default="")

    aggregate = sub.add_parser("aggregate", help="Write an aggregate status manifest.")
    aggregate.add_argument("--out", type=Path, required=True)
    aggregate.add_argument("--run-name", required=True)
    aggregate.add_argument("--status", required=True)
    aggregate.add_argument("--command-json")
    aggregate.add_argument("--input", action="append", default=[])
    aggregate.add_argument("--output", action="append", default=[])
    aggregate.add_argument("--method-manifest", type=Path, action="append", default=[])
    aggregate.add_argument("--metric", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = parse_label_path(args.input)
        outputs = parse_label_path(args.output)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.mode == "method":
        payload = build_external_method_manifest(
            method=args.method,
            command=parse_command(args.command_json),
            inputs=inputs,
            outputs=outputs,
            env=os.environ,
            status=args.status,
            returncode=args.return_code,
            runtime_seconds=args.runtime_seconds,
            message=args.message,
        )
    else:
        metrics: dict[str, str] = {}
        for item in args.metric:
            if "=" not in item:
                print(f"Error: Expected NAME=VALUE for --metric, got {item!r}", file=sys.stderr)
                return 2
            key, value = item.split("=", 1)
            metrics[key] = value
        payload = build_external_baseline_manifest(
            run_name=args.run_name,
            status=args.status,
            command=parse_command(args.command_json),
            inputs=inputs,
            outputs=outputs,
            env=os.environ,
            methods=load_method_manifests(args.method_manifest),
            metrics=metrics,
        )

    output = write_json(args.out, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for explicit native-format imports or local runs."""

import argparse
import json
from pathlib import Path

from .formats import parse_bbcontacts, parse_bcov, parse_betapro_map, parse_betapro_str
from .runner import run_bbcontacts, run_bcov, run_betapro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Run from a JSON specification matching a documented runner API")
    run.add_argument("tool", choices=["bcov", "bbcontacts", "betapro"])
    run.add_argument("--spec", type=Path, required=True)
    native = sub.add_parser("parse", help="Import previously completed native output")
    native.add_argument("format", choices=["bcov", "bbcontacts", "betapro-map", "betapro-str"])
    native.add_argument("--input", type=Path, required=True)
    native.add_argument("--sample-id", required=True)
    native.add_argument("--length", type=int)
    native.add_argument("--contacts", type=Path)
    native.add_argument("--strands", type=Path)
    native.add_argument("--successful-exit", action="store_true",
                        help="Attest independently verified successful exit; required for bbcontacts")
    native.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        runners = dict(bcov=run_bcov, bbcontacts=run_bbcontacts, betapro=run_betapro)
        result = runners[args.tool](**json.loads(args.spec.read_text()))
        print(json.dumps(dict(tool=result["tool"], status=result["status"],
                              sample_id=result["sample_id"])))
        return
    if args.format in ("bcov", "bbcontacts") and args.length is None:
        parser.error("--length is required for BCov and bbcontacts")
    if args.format == "bcov":
        if args.contacts is None or args.strands is None:
            parser.error("BCov requires --contacts and --strands")
        result = parse_bcov(args.input, args.contacts, args.strands,
                             sample_id=args.sample_id, length=args.length)
    elif args.format == "bbcontacts":
        result = parse_bbcontacts(args.input, sample_id=args.sample_id, length=args.length,
                                  successful_exit=args.successful_exit)
    else:
        reader = parse_betapro_map if args.format == "betapro-map" else parse_betapro_str
        result = reader(args.input, sample_id=args.sample_id)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

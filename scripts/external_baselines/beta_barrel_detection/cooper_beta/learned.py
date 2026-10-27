"""Score existing native Cooper-Beta graph features with a frozen fitted model."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path

FEATURES = ["strand_adjacency_count", "cycle_strand_count", "cycle_strand_fraction", "cycle_rank"]


def frozen_selection(metadata: dict, selection: str) -> float:
    if metadata["feature_columns"] != FEATURES:
        raise ValueError("Frozen model metadata does not declare the expected graph feature order")
    settings = metadata["machine_learning"]
    if selection == "deployment":
        threshold = float(settings["deployment_threshold"])
    elif selection.startswith("outer_"):
        fold = int(selection.removeprefix("outer_"))
        matches = [row for row in settings["outer_selections"] if row["outer_fold"] == fold]
        if len(matches) != 1:
            raise ValueError("Requested outer-fold selection is absent or ambiguous")
        threshold = float(matches[0]["threshold"])
    else:
        raise ValueError("Selection must be deployment or outer_N")
    if not 0 <= threshold <= 1:
        raise ValueError("Frozen probability threshold must lie in [0, 1]")
    return threshold


def score_native_result(native_result: dict, *, model_path: str | Path,
                        metadata_path: str | Path, selection: str = "deployment") -> dict:
    """Load a trusted local frozen artifact using its matching sklearn environment."""
    from importlib.metadata import version

    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    if native_result.get("status") != "ok" or native_result.get("method") != "cooper_beta":
        raise ValueError("A successful native Cooper-Beta feature result is required")
    row = native_result["native_result"]
    features = [float(row[name]) for name in FEATURES]
    if not all(math.isfinite(value) for value in features):
        raise ValueError("Frozen model features must be finite")
    metadata = json.loads(Path(metadata_path).read_text())
    threshold = frozen_selection(metadata, selection)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(model_path)
    if list(model.classes_) != [0, 1] or model.n_features_in_ != len(FEATURES):
        raise ValueError("Frozen model does not match the declared binary feature interface")
    score = float(model.predict_proba([features])[0, 1])
    if not 0 <= score <= 1:
        raise ValueError("Frozen model returned an invalid probability")
    return {"method": "cooper_beta_learned", "sample_id": native_result["sample_id"],
            "task": "barrel_geometry", "input_regime": "coordinates", "status": "ok",
            "score": score, "prediction": int(score >= threshold), "reason": "",
            "score_semantics": "frozen_model_positive_class_probability",
            "threshold": threshold, "selection": selection, "feature_columns": FEATURES,
            "features": features, "model_path": str(Path(model_path).resolve()),
            "metadata_path": str(Path(metadata_path).resolve()),
            "training_sample_count": metadata.get("sample_count"),
            "sklearn_version": version("scikit-learn"),
            "input": native_result.get("input"),
            "evaluation_note": "Frozen deployment scoring is not held-out evaluation on fitting members"}


def run_sample(native_result_path: str | Path, *, model_path: str | Path,
               metadata_path: str | Path, output_dir: str | Path,
               python_executable: str = sys.executable, selection: str = "deployment",
               timeout: float = 60.0) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"method": "cooper_beta_learned", "sample_id": "", "task": "barrel_geometry",
              "input_regime": "coordinates", "status": "unavailable", "score": None,
              "prediction": None, "reason": ""}
    try:
        native_path = Path(native_result_path).resolve()
        result["sample_id"] = json.loads(native_path.read_text())["sample_id"]
        command = [python_executable, "-B", str(Path(__file__).resolve()), "--worker",
                   str(native_path), "--model", str(Path(model_path).resolve()),
                   "--metadata", str(Path(metadata_path).resolve()), "--selection", selection]
        result["command"] = command
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                   check=False, cwd=output)
        (output / "stdout.txt").write_text(completed.stdout)
        (output / "stderr.txt").write_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"Frozen model worker failed with status {completed.returncode}; see stderr.txt")
        result.update(json.loads(completed.stdout))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        result.update(status="unavailable", score=None, prediction=None, reason=str(error))
    result["elapsed_seconds"] = time.monotonic() - started
    result["timing_scope"] = "frozen_model_import_load_and_scoring; native_graph_cost_recorded_separately"
    result["native_feature_result_path"] = str(Path(native_result_path).resolve())
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_result")
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--selection", default="deployment")
    parser.add_argument("--out-dir")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        result = score_native_result(json.loads(Path(args.native_result).read_text()),
                                     model_path=args.model, metadata_path=args.metadata,
                                     selection=args.selection)
    else:
        if not args.out_dir:
            parser.error("--out-dir is required")
        result = run_sample(args.native_result, model_path=args.model, metadata_path=args.metadata,
                            selection=args.selection, output_dir=args.out_dir,
                            python_executable=args.python, timeout=args.timeout)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

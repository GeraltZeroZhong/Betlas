"""Isolated SOLeNNoID CPU worker; import the user's upstream source unchanged."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path


def legacy_groups_one_config(config: dict) -> dict:
    """Remove only the legacy no-op groups=1 keyword, rejecting other semantics."""
    config = dict(config)
    if config.pop("groups", 1) != 1:
        raise ValueError("Only legacy Conv2DTranspose groups=1 is compatible")
    return config


class NativePredictor:
    """Reuse one official model while keeping the native per-structure algorithm.

    Callers own process isolation, time limits, output persistence and resource
    limits. No coordinates, model parameters, scores or upstream steps change.
    """

    def __init__(self, source_dir, model_path, *, keras3_groups_one_compat=False):
        started = time.monotonic()
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import numpy as np
        import tensorflow as tf

        shim = "float" not in np.__dict__
        if shim:
            np.float = float
        sys.path.insert(0, str(Path(source_dir).resolve()))
        self.upstream = importlib.import_module("solenoid_detector_onehot_v3")
        custom_objects = None
        if keras3_groups_one_compat:
            class LegacyConv2DTranspose(tf.keras.layers.Conv2DTranspose):
                def __init__(self, **kwargs):
                    super().__init__(**legacy_groups_one_config(kwargs))
            custom_objects = {"Conv2DTranspose": LegacyConv2DTranspose}
        load_started = time.monotonic()
        self.model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        self.metadata = {
            "numpy_version": np.__version__, "tensorflow_version": tf.__version__,
            "numpy_float_builtin_alias_shim": shim, "model_load_compile": False,
            "keras3_groups_one_compat": keras3_groups_one_compat,
            "source_dir": str(Path(source_dir).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "score_channels": "native_first_four_of_six_no_renormalization",
            "execution_device": "CPU", "model_parameters": int(self.model.count_params()),
            "model_load_seconds": time.monotonic() - load_started,
            "predictor_setup_seconds": time.monotonic() - started,
        }

    def predict(self, inspection: dict, output_dir) -> dict:
        """Write the same native span table as the isolated single-case worker."""
        from Bio.PDB import MMCIFParser, PDBParser

        started = time.monotonic()
        if inspection["status"] != "ready":
            raise ValueError("Structure inspection is not ready")
        path = inspection["structure_path"]
        reader = MMCIFParser(QUIET=True) if Path(path).suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True, PERMISSIVE=False)
        structure = reader.get_structure("input", path)
        protein = self.upstream.Protein(path, chain=inspection["chain_id"], structure=structure)
        protein.set_up()
        first, matrix, _ = protein.return_matrix_and_first_res()
        expected_numbers = [x["residue_number"] for x in inspection["residues"]]
        if list(protein.indices) != expected_numbers or first != expected_numbers[0]:
            raise ValueError("Upstream residue indexing differs from inspected mapping")
        if matrix.shape != (inspection["matrix_span"], inspection["matrix_span"]):
            raise ValueError("Upstream matrix shape differs from inspected span")
        prediction = self.upstream.Prediction(
            first_res_chain=first, distance_matrix_chain=matrix, loaded_model=self.model,
            sub_matrix_size=128, maxlength=protein.indices[-1] - first,
        )
        inference_started = time.monotonic()
        scores = prediction.get_final_onehot()
        inference_seconds = time.monotonic() - inference_started
        output = Path(output_dir)
        with (output / "native_span_scores.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["span_index", "score_0", "score_1", "score_2", "score_3"])
            writer.writerows([index, *[float(x) for x in row]] for index, row in enumerate(scores))
        metadata = dict(self.metadata, inference_seconds=inference_seconds,
                        worker_elapsed_seconds=time.monotonic() - started,
                        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        model_reuse=True)
        (output / "worker.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keras3-groups-one-compat", action="store_true")
    args = parser.parse_args()
    inspection = json.loads(Path(args.inspection).read_text())
    if inspection["status"] != "ready":
        raise ValueError("Structure inspection is not ready")
    predictor = NativePredictor(args.source_dir, args.model_path,
                                keras3_groups_one_compat=args.keras3_groups_one_compat)
    metadata = predictor.predict(inspection, args.output_dir)
    metadata.update(worker_elapsed_seconds=time.monotonic() - started, model_reuse=False)
    (Path(args.output_dir) / "worker.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

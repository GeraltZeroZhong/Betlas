"""Import saved native RAPHAEL HTML results. No unverified local CLI or upload."""
from __future__ import annotations

import argparse
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def parse_native_html(path: str | Path, *, sample_id: str) -> dict:
    """Read SVM SCORE (the decision score), distinct from Total score.

    The caller associates this saved result with the exact submitted structure;
    a PDB code in a web result alone does not establish crop or coordinate identity.
    """
    parser = _Text()
    parser.feed(Path(path).read_text(encoding="utf-8"))
    text = " ".join(parser.parts)
    status = re.search(r"\bStatus:\s*(\w+)", text, flags=re.I)
    if not status or status.group(1).lower() != "finished":
        raise ValueError("RAPHAEL native result is not marked finished")
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    scores = re.findall(r"\bSVM\s+SCORE\s*:\s*" + number, text, flags=re.I)
    if len(scores) != 1:
        raise ValueError("Expected exactly one native RAPHAEL SVM SCORE")
    score = float(scores[0])
    if not math.isfinite(score):
        raise ValueError("Nonfinite RAPHAEL score")
    period_match = re.search(r"Predicted\s+repeat\s+length\s*:\s*" + number, text, flags=re.I)
    source = re.search(r"\bPDB\s*:\s*(\S+)", text)
    return {
        "sample_id": sample_id, "method": "RAPHAEL", "status": "ok",
        "execution_mode": "saved_native_result_import",
        "prediction_execution_verified": False,
        "svm_score": score, "score_is_probability": False,
        "native_decision": "repeat" if score > 0 else "globular" if score < 0 else "boundary",
        "predicted_repeat_length": float(period_match.group(1)) if period_match else None,
        "native_structure_identifier": source.group(1) if source else None,
        "input_identity_status": "caller_must_verify_exact_submitted_coordinates_and_crop",
        "source_result": str(Path(path).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-html", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = parse_native_html(args.native_html, sample_id=args.sample_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

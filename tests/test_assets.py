from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from betlas import cli
from betlas.assets import (
    AssetError,
    describe_asset,
    download_asset,
    list_assets,
    resolve_asset_path,
    resolve_esmc_weights,
    verify_asset,
)
from scripts.reproducibility.readouts.beta_barrel_detection.betlas_readout import (
    ReadoutPaths,
    build_or_load_betlas_151,
    build_or_load_layer_radial16,
    load_esmc_embeddings,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    root: Path,
    *,
    asset_id: str,
    filename: str,
    data: bytes,
    byte_size: int | None = None,
    sha256: str | None = None,
    release_status: str = "test",
    download_path: str | None = None,
) -> Path:
    manifest_dir = root / "manifests" / "example" / "official"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                f"asset_id: {asset_id}",
                "asset_type: unit_bundle",
                "readout: beta_barrel_detection",
                "profile: unit",
                f"release_status: {release_status}",
                "bundle: unit.zip",
                "bundle_subdir: example/official",
                "generated_by:",
                "  - tests/test_assets.py",
                "source_inputs:",
                "  - unit",
                "files:",
                f"  - filename: {filename}",
                "    purpose: unit fixture",
                f"    byte_size: {len(data) if byte_size is None else byte_size}",
                f"    sha256: {sha256 or _sha256(data)}",
                f"    download_path: {download_path or f'example/official/{filename}'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    release_file = root / "release" / "example" / "official" / filename
    release_file.parent.mkdir(parents=True)
    release_file.write_bytes(data)
    return manifest_dir.parent.parent


def test_asset_download_verify_and_path_use_local_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"betlas asset fixture\n"
    manifest_root = _write_manifest(tmp_path, asset_id="unit-asset", filename="tiny.txt", data=data)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))
    monkeypatch.setenv("BETLAS_ASSET_BASE_URL", str(tmp_path / "release"))

    assert list_assets() == ("unit-asset",)
    assert describe_asset("unit-asset")["files"][0]["filename"] == "tiny.txt"

    paths = download_asset("unit-asset", cache_dir=cache_dir)

    assert paths == (cache_dir / "unit-asset" / "tiny.txt",)
    assert paths[0].read_bytes() == data
    assert verify_asset("unit-asset", cache_dir=cache_dir) == {"tiny.txt": True}
    assert resolve_asset_path("unit-asset", "tiny.txt", cache_dir=cache_dir) == paths[0]


def test_asset_download_rejects_hash_and_size_mismatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="unit-asset",
        filename="tiny.txt",
        data=b"abc",
        byte_size=4,
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))
    monkeypatch.setenv("BETLAS_ASSET_BASE_URL", str(tmp_path / "release"))

    with pytest.raises(AssetError, match="size mismatch"):
        download_asset("unit-asset", cache_dir=tmp_path / "cache")

    manifest_root = _write_manifest(
        tmp_path / "sha",
        asset_id="unit-asset-sha",
        filename="tiny.txt",
        data=b"abc",
        sha256=_sha256(b"abd"),
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))
    monkeypatch.setenv("BETLAS_ASSET_BASE_URL", str(tmp_path / "sha" / "release"))

    with pytest.raises(AssetError, match="sha256 mismatch"):
        download_asset("unit-asset-sha", cache_dir=tmp_path / "cache")


def test_asset_manifest_rejects_unsafe_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="unsafe-asset",
        filename="safe.txt",
        data=b"bad",
    )
    manifest = next(manifest_root.rglob("manifest.yaml"))
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("filename: safe.txt", "filename: ../escape.txt"),
        encoding="utf-8",
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))

    with pytest.raises(AssetError, match="unsafe"):
        list_assets()


def test_asset_manifest_rejects_unsafe_asset_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="../escaped-cache-root",
        filename="safe.txt",
        data=b"bad",
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))

    with pytest.raises(AssetError, match="asset_id is unsafe"):
        list_assets()


def test_resolve_asset_path_validates_asset_and_filename_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"betlas asset fixture\n"
    manifest_root = _write_manifest(tmp_path, asset_id="unit-asset", filename="tiny.txt", data=data)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))

    assert resolve_asset_path("unit-asset", "tiny.txt", cache_dir=cache_dir) == cache_dir / "unit-asset" / "tiny.txt"
    with pytest.raises(AssetError, match="unknown Betlas asset"):
        resolve_asset_path("missing-asset", "tiny.txt", cache_dir=cache_dir)
    with pytest.raises(AssetError, match="does not contain"):
        resolve_asset_path("unit-asset", "other.txt", cache_dir=cache_dir)
    with pytest.raises(AssetError, match="does not contain"):
        resolve_asset_path("unit-asset", "../escape.txt", cache_dir=cache_dir)


def test_asset_download_force_replaces_bad_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"correct\n"
    manifest_root = _write_manifest(tmp_path, asset_id="unit-asset", filename="tiny.txt", data=data)
    cache_dir = tmp_path / "cache"
    target = cache_dir / "unit-asset" / "tiny.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))
    monkeypatch.setenv("BETLAS_ASSET_BASE_URL", str(tmp_path / "release"))

    with pytest.raises(AssetError, match="use force=True"):
        download_asset("unit-asset", cache_dir=cache_dir)

    assert download_asset("unit-asset", cache_dir=cache_dir, force=True) == (target,)
    assert target.read_bytes() == data


def test_pending_asset_download_requires_explicit_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"pending\n"
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="pending-asset",
        filename="tiny.txt",
        data=data,
        release_status="pending_release",
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))
    monkeypatch.delenv("BETLAS_ASSET_BASE_URL", raising=False)

    with pytest.raises(AssetError, match="pending_release"):
        download_asset("pending-asset", cache_dir=tmp_path / "cache")

    monkeypatch.setenv("BETLAS_ASSET_BASE_URL", str(tmp_path / "release"))
    assert download_asset("pending-asset", cache_dir=tmp_path / "cache")[0].read_bytes() == data


def test_asset_manifest_rejects_unsafe_download_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="unsafe-download-path",
        filename="tiny.txt",
        data=b"bad",
        download_path="../escape/tiny.txt",
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))

    with pytest.raises(AssetError, match="download_path is unsafe"):
        list_assets()


def test_asset_manifest_validates_required_file_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="missing-sha",
        filename="tiny.txt",
        data=b"bad",
    )
    manifest = next(manifest_root.rglob("manifest.yaml"))
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("    sha256: " + _sha256(b"bad") + "\n", ""),
        encoding="utf-8",
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))

    with pytest.raises(AssetError, match="missing sha256"):
        list_assets()


def test_asset_manifest_validates_sha256_and_byte_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = _write_manifest(
        tmp_path,
        asset_id="invalid-sha",
        filename="tiny.txt",
        data=b"bad",
        sha256="not-a-sha",
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))

    with pytest.raises(AssetError, match="invalid sha256"):
        list_assets()

    manifest_root = _write_manifest(
        tmp_path / "size",
        asset_id="invalid-size",
        filename="tiny.txt",
        data=b"bad",
        byte_size=0,
    )
    monkeypatch.setenv("BETLAS_ASSET_MANIFEST_DIR", str(manifest_root))
    with pytest.raises(AssetError, match="non-positive byte_size"):
        list_assets()


def test_esmc_npz_loader_uses_non_pickle_string_ids(tmp_path: Path) -> None:
    npz = tmp_path / "esmc.npz"
    import numpy as np

    np.savez_compressed(
        npz,
        record_id=np.asarray(["r1", "r2"], dtype=str),
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        esmc_available=np.asarray([1.0, 0.0], dtype=float),
    )

    embeddings, available = load_esmc_embeddings(npz, ["r2", "r1"])

    assert embeddings.index.tolist() == ["r2", "r1"]
    assert available.tolist() == [0.0, 1.0]

    unsafe = tmp_path / "unsafe.npz"
    np.savez_compressed(
        unsafe,
        record_id=np.asarray(["r1"], dtype=object),
        embeddings=np.asarray([[1.0]], dtype=float),
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_esmc_embeddings(unsafe, ["r1"])


def test_resolve_esmc_weights_only_checks_local_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETLAS_ESMC_WEIGHTS", raising=False)

    assert resolve_esmc_weights() is None
    with pytest.raises(AssetError, match="not provided"):
        resolve_esmc_weights(required=True)
    with pytest.raises(AssetError, match="does not exist"):
        resolve_esmc_weights(tmp_path / "missing.pt")

    weights = tmp_path / "esmc.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("BETLAS_ESMC_WEIGHTS", str(weights))
    assert resolve_esmc_weights() == weights


def test_assets_cli_help_and_json_describe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["assets", "--help"])
    assert exc.value.code == 0
    assert "download" in capsys.readouterr().out

    cli.main(["assets", "describe", "betlas-beta-barrel-staves-official-v1", "--format", "json"])
    out = capsys.readouterr().out
    assert "betlas-beta-barrel-staves-official-v1" in out
    assert "betlas_151_chain_features.csv" in out

    cli.main(["assets", "describe", "betlas-beta-barrel-staves-official-v1"])
    text = capsys.readouterr().out
    assert "Release status:" in text
    assert "purpose=" in text


def test_staves_official_manifest_covers_runner_required_inputs() -> None:
    manifest = describe_asset("betlas-beta-barrel-staves-official-v1")
    filenames = {str(item["filename"]) for item in manifest["files"]}
    assert {
        "per_record_aligned_wide.csv",
        "feature_columns_151.csv",
        "betlas_151_chain_features.csv",
        "layer_radial16_feature_values.csv",
        "esmc_mean_embeddings_aligned.npz",
        "esmc_embedding_coverage.csv",
    } <= filenames


def test_detection_official_manifest_covers_fixed_runner_inputs() -> None:
    manifest = describe_asset("betlas-beta-barrel-detection-official-v1")
    filenames = {str(item["filename"]) for item in manifest["files"]}
    assert {
        "benchmark_cohort.csv",
        "feature_columns_151.csv",
        "betlas_151_chain_features.csv",
        "layer_radial16_feature_manifest.csv",
        "layer_radial16_feature_values.csv",
        "esmc_mean_embeddings_aligned.npz",
        "official_fold_metrics.csv",
        "official_per_record_predictions.csv",
    } <= filenames
    by_name = {str(item["filename"]): item for item in manifest["files"]}
    assert (
        by_name["betlas_151_chain_features.csv"]["download_path"]
        == "beta_barrel_detection/official/betlas_151_chain_features.csv"
    )


def test_detection_fixed_runner_consumes_cached_feature_and_layer_inputs(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        [
            {"record_id": "r1", "filename": "one.cif"},
            {"record_id": "r2", "filename": "two.cif"},
        ]
    )
    features_csv = tmp_path / "betlas_151_chain_features.csv"
    layer_values_csv = tmp_path / "layer_radial16_feature_values.csv"
    layer_manifest_csv = tmp_path / "layer_radial16_feature_manifest.csv"
    pd.DataFrame(
        [
            {"record_id": "r2", "betlas_axis_best_angular_coverage": 0.2},
            {"record_id": "r1", "betlas_axis_best_angular_coverage": 0.1},
        ]
    ).to_csv(features_csv, index=False)
    pd.DataFrame(
        [
            {"record_id": "r2", "layer_radial16__decision_score": 0.8},
            {"record_id": "r1", "layer_radial16__decision_score": 0.7},
        ]
    ).to_csv(layer_values_csv, index=False)
    pd.DataFrame([{"feature": "layer_radial16__decision_score"}]).to_csv(layer_manifest_csv, index=False)
    paths = ReadoutPaths(
        out_dir=tmp_path / "out",
        features_csv=features_csv,
        layer_values_csv=layer_values_csv,
        layer_manifest_csv=layer_manifest_csv,
    )

    features = build_or_load_betlas_151(
        paths=paths,
        cohort=cohort,
        feature_columns=["betlas_axis_best_angular_coverage"],
    )
    layer_values, layer_manifest = build_or_load_layer_radial16(paths=paths, cohort=cohort)

    assert features["record_id"].tolist() == ["r1", "r2"]
    assert layer_values["record_id"].tolist() == ["r1", "r2"]
    assert layer_manifest["feature"].tolist() == ["layer_radial16__decision_score"]
    assert (paths.out_dir / "betlas_151_chain_features.csv").exists()
    assert (paths.out_dir / "layer_radial16_feature_values.csv").exists()

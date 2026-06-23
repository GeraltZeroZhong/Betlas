from __future__ import annotations

import hashlib
import os
import posixpath
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ASSET_BASE_URL = "https://github.com/GeraltZeroZhong/Betlas/releases/download/v1.0.0/"
ASSET_BASE_URL_ENV = "BETLAS_ASSET_BASE_URL"
ASSET_DIR_ENV = "BETLAS_ASSET_DIR"
ESMC_WEIGHTS_ENV = "BETLAS_ESMC_WEIGHTS"
_ASSET_MANIFEST_DIR_ENV = "BETLAS_ASSET_MANIFEST_DIR"
_PACKAGE_MANIFEST_ROOT = "asset_manifests"
_URL_TIMEOUT_SECONDS = 30
_PENDING_RELEASE_STATUS = "pending_release"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SUPPORTED_SCHEMA_VERSION = "betlas.asset-manifest.v1"
_REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "asset_id",
    "asset_type",
    "readout",
    "profile",
    "release_status",
    "bundle",
    "bundle_subdir",
    "generated_by",
    "source_inputs",
    "files",
)
_REQUIRED_FILE_FIELDS = (
    "filename",
    "purpose",
    "byte_size",
    "sha256",
    "download_path",
)


class AssetError(RuntimeError):
    """Raised when a Betlas asset cannot be resolved, downloaded, or verified."""


def _cache_dir() -> Path:
    return Path(os.environ.get(ASSET_DIR_ENV, "~/.cache/betlas/assets")).expanduser()


def _iter_package_manifest_paths() -> Iterable[Any]:
    root = resources.files("betlas").joinpath(_PACKAGE_MANIFEST_ROOT)

    def walk(node: Any) -> Iterable[Any]:
        if node.is_file() and node.name == "manifest.yaml":
            yield node
            return
        if node.is_dir():
            for child in sorted(node.iterdir(), key=lambda item: item.name):
                yield from walk(child)

    yield from walk(root)


def _iter_manifest_paths() -> Iterable[Any]:
    override = os.environ.get(_ASSET_MANIFEST_DIR_ENV)
    if override:
        yield from sorted(Path(override).expanduser().rglob("manifest.yaml"))
        return
    yield from _iter_package_manifest_paths()


def _read_yaml(path: Any) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise AssetError(f"asset manifest is not a mapping: {path}")
    return data


def _manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise AssetError(f"asset manifest {manifest.get('asset_id', '<unknown>')} has invalid files")
    out: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise AssetError(f"asset manifest {manifest.get('asset_id', '<unknown>')} has non-mapping file entry")
        file_info = dict(item)
        for field in _REQUIRED_FILE_FIELDS:
            if field not in file_info or str(file_info.get(field, "")).strip() == "":
                raise AssetError(
                    f"asset manifest {manifest.get('asset_id', '<unknown>')} file entry missing {field}"
                )
        _validate_asset_filename(str(file_info.get("filename", "")))
        file_info["download_path"] = _validate_download_path(str(file_info["download_path"]))
        try:
            byte_size = int(file_info["byte_size"])
        except (TypeError, ValueError) as exc:
            raise AssetError(
                f"asset manifest {manifest.get('asset_id', '<unknown>')} file has invalid byte_size"
            ) from exc
        if byte_size <= 0:
            raise AssetError(
                f"asset manifest {manifest.get('asset_id', '<unknown>')} file has non-positive byte_size"
            )
        file_info["byte_size"] = byte_size
        sha256 = str(file_info["sha256"]).strip()
        if not _SHA256_RE.fullmatch(sha256):
            raise AssetError(
                f"asset manifest {manifest.get('asset_id', '<unknown>')} file has invalid sha256"
            )
        file_info["sha256"] = sha256.lower()
        out.append(file_info)
    return out


def _validate_asset_filename(filename: str) -> str:
    name = str(filename).strip()
    path = Path(name)
    if not name:
        raise AssetError("asset manifest contains an empty filename")
    if path.is_absolute() or ".." in path.parts:
        raise AssetError(f"asset filename is unsafe: {filename!r}")
    if len(path.parts) != 1:
        raise AssetError(f"asset filename must not contain directories: {filename!r}")
    return name


def _validate_asset_id(asset_id: str) -> str:
    value = str(asset_id).strip()
    path = Path(value)
    if not value:
        raise AssetError("asset_id must not be empty")
    if path.is_absolute() or ".." in path.parts:
        raise AssetError(f"asset_id is unsafe: {asset_id!r}")
    if len(path.parts) != 1 or "/" in value or "\\" in value:
        raise AssetError(f"asset_id must not contain directories: {asset_id!r}")
    return value


def _validate_download_path(download_path: str) -> str:
    path = str(download_path).strip()
    if not path:
        raise AssetError("asset manifest contains an empty download_path")
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme or parsed.netloc:
        raise AssetError(f"asset download_path must be release-relative: {download_path!r}")
    if path.startswith("/"):
        raise AssetError(f"asset download_path must not be absolute: {download_path!r}")
    raw_parts = path.replace("\\", "/").split("/")
    if ".." in raw_parts:
        raise AssetError(f"asset download_path is unsafe: {download_path!r}")
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized in {"", "."} or normalized == ".." or normalized.startswith("../"):
        raise AssetError(f"asset download_path is unsafe: {download_path!r}")
    return normalized


def _validate_non_empty_string_list(value: Any, *, field: str, asset_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AssetError(f"asset manifest {asset_id} field {field} must be a non-empty list")
    values = [str(item).strip() for item in value]
    if any(not item for item in values):
        raise AssetError(f"asset manifest {asset_id} field {field} contains an empty value")
    return values


def _validate_manifest_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    asset_id = _validate_asset_id(str(manifest.get("asset_id", "")).strip())
    schema_version = str(manifest.get("schema_version", "")).strip()
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise AssetError(
            f"asset manifest {asset_id} has unsupported schema_version {schema_version!r}; "
            f"expected {_SUPPORTED_SCHEMA_VERSION!r}"
        )
    manifest["asset_id"] = asset_id
    manifest["schema_version"] = schema_version
    manifest["bundle"] = _validate_download_path(str(manifest.get("bundle", "")))
    bundle_subdir = _validate_download_path(str(manifest.get("bundle_subdir", "")))
    manifest["bundle_subdir"] = bundle_subdir
    manifest["generated_by"] = _validate_non_empty_string_list(
        manifest.get("generated_by"),
        field="generated_by",
        asset_id=asset_id,
    )
    manifest["source_inputs"] = _validate_non_empty_string_list(
        manifest.get("source_inputs"),
        field="source_inputs",
        asset_id=asset_id,
    )
    manifest["files"] = _manifest_files(manifest)

    filenames: set[str] = set()
    download_paths: set[str] = set()
    required_prefix = bundle_subdir.rstrip("/") + "/"
    for file_info in manifest["files"]:
        filename = str(file_info["filename"])
        download_path = str(file_info["download_path"])
        if filename in filenames:
            raise AssetError(f"asset manifest {asset_id} contains duplicate filename {filename!r}")
        if download_path in download_paths:
            raise AssetError(f"asset manifest {asset_id} contains duplicate download_path {download_path!r}")
        if not download_path.startswith(required_prefix):
            raise AssetError(
                f"asset manifest {asset_id} download_path {download_path!r} does not start with "
                f"bundle_subdir {bundle_subdir!r}"
            )
        filenames.add(filename)
        download_paths.add(download_path)
    return manifest


def _load_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in _iter_manifest_paths():
        manifest = _read_yaml(path)
        for field in _REQUIRED_TOP_LEVEL_FIELDS:
            if field not in manifest or str(manifest.get(field, "")).strip() == "":
                raise AssetError(f"asset manifest is missing {field}: {path}")
        manifest = _validate_manifest_contract(manifest)
        asset_id = str(manifest["asset_id"])
        if asset_id in manifests:
            raise AssetError(f"duplicate asset_id in manifests: {asset_id}")
        manifests[asset_id] = manifest
    return manifests


def _get_manifest(asset_id: str) -> dict[str, Any]:
    manifests = _load_manifests()
    try:
        return manifests[asset_id]
    except KeyError as exc:
        available = ", ".join(sorted(manifests))
        raise AssetError(f"unknown Betlas asset {asset_id!r}; available assets: {available}") from exc


def _select_files(manifest: dict[str, Any], filenames: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = _manifest_files(manifest)
    if filenames is None:
        return selected
    requested = {str(name) for name in filenames}
    by_name = {str(item["filename"]): item for item in selected}
    missing = sorted(requested - set(by_name))
    if missing:
        raise AssetError(f"asset {manifest['asset_id']} does not contain file(s): {missing}")
    return [by_name[name] for name in sorted(requested)]


def _asset_root(asset_id: str, cache_dir: str | Path | None = None) -> Path:
    root = Path(cache_dir).expanduser() if cache_dir is not None else _cache_dir()
    return root / _validate_asset_id(asset_id)


def _asset_file_path(root: Path, file_info: dict[str, Any]) -> Path:
    return root / _validate_asset_filename(str(file_info["filename"]))


def _base_url(value: str | None = None) -> str:
    return str(value or os.environ.get(ASSET_BASE_URL_ENV, DEFAULT_ASSET_BASE_URL))


def _download_source(file_info: dict[str, Any], base_url: str | None = None) -> str:
    path = str(file_info.get("download_path") or file_info.get("download_uri") or file_info["filename"])
    if path.startswith("release://"):
        path = path.split("/", 3)[-1]
    path = _validate_download_path(path)

    base = _base_url(base_url)
    base_parsed = urllib.parse.urlparse(base)
    if base_parsed.scheme in {"http", "https", "file"}:
        return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    return str(Path(base).expanduser() / path)


def _bundle_source(manifest: dict[str, Any], base_url: str | None = None) -> str:
    bundle = _validate_download_path(str(manifest["bundle"]))
    base = _base_url(base_url)
    base_parsed = urllib.parse.urlparse(base)
    if base_parsed.scheme in {"http", "https", "file"}:
        return urllib.parse.urljoin(base.rstrip("/") + "/", bundle.lstrip("/"))
    return str(Path(base).expanduser() / bundle)


def _using_explicit_base_url(base_url: str | None = None) -> bool:
    return base_url is not None or ASSET_BASE_URL_ENV in os.environ


def _require_released_or_explicit_mirror(manifest: dict[str, Any], base_url: str | None = None) -> None:
    status = str(manifest.get("release_status", "")).strip().lower()
    if status == _PENDING_RELEASE_STATUS and not _using_explicit_base_url(base_url):
        asset_id = str(manifest.get("asset_id", "<unknown>"))
        raise AssetError(
            f"asset {asset_id} is marked pending_release; the default Betlas release payload is not "
            f"available yet. Set {ASSET_BASE_URL_ENV} or pass --base-url to a local mirror that follows "
            "the manifest download_path layout."
        )


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _verify_file(path: Path, file_info: dict[str, Any]) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    digest, size = _sha256_and_size(path)
    expected_size = int(file_info["byte_size"])
    expected_sha = str(file_info["sha256"]).lower()
    if size != expected_size:
        return False, f"size mismatch: expected {expected_size}, observed {size}"
    if digest.lower() != expected_sha:
        return False, f"sha256 mismatch: expected {expected_sha}, observed {digest}"
    return True, "ok"


def _copy_source_to_path(source: str, target: Path) -> None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https", "file"}:
        with urllib.request.urlopen(source, timeout=_URL_TIMEOUT_SECONDS) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return
    with Path(source).expanduser().open("rb") as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _copy_from_source(source: str, target: Path) -> None:
    try:
        _copy_source_to_path(source, target)
    except Exception as exc:  # pragma: no cover - exact network errors vary by platform
        raise AssetError(f"could not copy asset file from {source!r}: {exc}") from exc


def _materialize_zip_bundle(bundle_source: str, target_dir: Path) -> Path:
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(target_dir), prefix=".bundle.", suffix=".zip") as handle:
            bundle_tmp = Path(handle.name)
        _copy_source_to_path(bundle_source, bundle_tmp)
    except Exception as exc:  # pragma: no cover - exact network/zip errors vary by platform
        raise AssetError(f"could not copy asset bundle from {bundle_source!r}: {exc}") from exc
    return bundle_tmp


def _copy_from_zip_bundle_file(bundle_path: Path, member_path: str, target: Path) -> None:
    member = _validate_download_path(member_path)
    try:
        with zipfile.ZipFile(bundle_path) as zf:
            try:
                with zf.open(member) as response, target.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            except KeyError as exc:
                raise AssetError(f"bundle {bundle_path!s} does not contain {member!r}") from exc
    except AssetError:
        raise
    except Exception as exc:  # pragma: no cover - exact zip errors vary by platform
        raise AssetError(f"could not copy asset file {member!r} from bundle {bundle_path!s}: {exc}") from exc


def list_assets() -> tuple[str, ...]:
    """Return available Betlas asset bundle identifiers."""

    return tuple(sorted(_load_manifests()))


def describe_asset(asset_id: str) -> dict[str, Any]:
    """Return the manifest dictionary for a Betlas asset bundle."""

    return dict(_get_manifest(asset_id))


def resolve_asset_path(
    asset_id: str,
    filename: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    download: bool = False,
    force: bool = False,
    base_url: str | None = None,
) -> Path:
    """Return the local cache path for an asset bundle or one file in it."""

    manifest = _get_manifest(asset_id)
    selected_files = _select_files(manifest, [filename] if filename else None)
    if download:
        download_asset(
            asset_id,
            cache_dir=cache_dir,
            filenames=[filename] if filename else None,
            force=force,
            base_url=base_url,
        )
    root = _asset_root(asset_id, cache_dir)
    if filename is None:
        return root
    return _asset_file_path(root, selected_files[0])


def verify_asset(
    asset_id: str,
    *,
    cache_dir: str | Path | None = None,
    filenames: Iterable[str] | None = None,
    strict: bool = False,
) -> dict[str, bool]:
    """Verify cached asset files against manifest SHA-256 hashes and byte sizes."""

    manifest = _get_manifest(asset_id)
    root = _asset_root(asset_id, cache_dir)
    results: dict[str, bool] = {}
    errors: list[str] = []
    for file_info in _select_files(manifest, filenames):
        filename = str(file_info["filename"])
        ok, reason = _verify_file(_asset_file_path(root, file_info), file_info)
        results[filename] = ok
        if not ok:
            hint = f"cache={root}"
            if str(manifest.get("release_status", "")).strip().lower() == _PENDING_RELEASE_STATUS:
                hint += (
                    f"; asset is pending_release, populate the cache with `betlas assets download "
                    f"{manifest.get('asset_id', '<asset>')} --base-url <local mirror>` or set "
                    f"{ASSET_BASE_URL_ENV} for that download first"
                )
            errors.append(f"{filename}: {reason} ({hint})")
    if strict and errors:
        if len(errors) > 1 and str(manifest.get("release_status", "")).strip().lower() == _PENDING_RELEASE_STATUS:
            first_errors = "; ".join(error.split(" (", 1)[0] for error in errors[:3])
            more = f"; plus {len(errors) - 3} more file(s)" if len(errors) > 3 else ""
            raise AssetError(
                f"asset {asset_id} verification failed for {len(errors)} file(s) in cache={root}; "
                "asset is pending_release, populate the cache with "
                f"`betlas assets download {manifest.get('asset_id', '<asset>')} --base-url <local mirror>` "
                f"or set {ASSET_BASE_URL_ENV} for that download first. First failures: "
                f"{first_errors}{more}"
            )
        raise AssetError("; ".join(errors))
    return results


def asset_file_report(
    asset_id: str,
    *,
    cache_dir: str | Path | None = None,
    filenames: Iterable[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Return expected and observed cache state for selected asset files."""

    manifest = _get_manifest(asset_id)
    root = _asset_root(asset_id, cache_dir)
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for file_info in _select_files(manifest, filenames):
        filename = str(file_info["filename"])
        path = _asset_file_path(root, file_info)
        ok, reason = _verify_file(path, file_info)
        observed_sha256 = ""
        observed_byte_size = 0
        if path.exists():
            observed_sha256, observed_byte_size = _sha256_and_size(path)
        if not ok:
            errors.append(f"{filename}: {reason}")
        files.append(
            {
                "filename": filename,
                "download_path": str(file_info["download_path"]),
                "path": str(path),
                "ok": bool(ok),
                "reason": reason,
                "expected_sha256": str(file_info["sha256"]),
                "expected_byte_size": int(file_info["byte_size"]),
                "observed_sha256": observed_sha256,
                "observed_byte_size": int(observed_byte_size),
            }
        )
    if strict and errors:
        if str(manifest.get("release_status", "")).strip().lower() == _PENDING_RELEASE_STATUS:
            first_errors = "; ".join(errors[:3])
            more = f"; plus {len(errors) - 3} more file(s)" if len(errors) > 3 else ""
            raise AssetError(
                f"asset {asset_id} verification failed for {len(errors)} file(s); "
                "asset is pending_release, populate the cache with "
                f"`betlas assets download {manifest.get('asset_id', '<asset>')} --base-url <local mirror>` "
                f"or set {ASSET_BASE_URL_ENV} for that download first. First failures: "
                f"{first_errors}{more}"
            )
        raise AssetError("; ".join(errors))
    return {
        "schema_version": str(manifest["schema_version"]),
        "asset_id": str(manifest["asset_id"]),
        "asset_type": str(manifest["asset_type"]),
        "readout": str(manifest["readout"]),
        "profile": str(manifest["profile"]),
        "release_status": str(manifest["release_status"]),
        "bundle": str(manifest["bundle"]),
        "bundle_subdir": str(manifest["bundle_subdir"]),
        "files": files,
    }


def download_asset(
    asset_id: str,
    *,
    cache_dir: str | Path | None = None,
    filenames: Iterable[str] | None = None,
    force: bool = False,
    base_url: str | None = None,
) -> tuple[Path, ...]:
    """Download an asset bundle into the Betlas cache and verify every file."""

    manifest = _get_manifest(asset_id)
    _require_released_or_explicit_mirror(manifest, base_url=base_url)
    root = _asset_root(asset_id, cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    bundle_tmp: Path | None = None
    bundle_source: str | None = None
    try:
        for file_info in _select_files(manifest, filenames):
            target = _asset_file_path(root, file_info)
            if target.exists() and not force:
                ok, reason = _verify_file(target, file_info)
                if ok:
                    downloaded.append(target)
                    continue
                raise AssetError(f"cached file failed verification ({target}): {reason}; use force=True to replace it")

            source = _download_source(file_info, base_url=base_url)
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=str(root),
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                ) as handle:
                    tmp_path = Path(handle.name)
                try:
                    _copy_from_source(source, tmp_path)
                except AssetError as direct_exc:
                    bundle_source = bundle_source or _bundle_source(manifest, base_url=base_url)
                    if bundle_tmp is None:
                        bundle_tmp = _materialize_zip_bundle(bundle_source, root)
                    try:
                        _copy_from_zip_bundle_file(bundle_tmp, str(file_info["download_path"]), tmp_path)
                    except AssetError as bundle_exc:
                        raise AssetError(
                            f"could not download {target.name!r} from {source!r} or bundle "
                            f"{bundle_source!r}: direct={direct_exc}; bundle={bundle_exc}"
                        ) from bundle_exc
                ok, reason = _verify_file(tmp_path, file_info)
                if not ok:
                    raise AssetError(f"downloaded file failed verification ({target.name}): {reason}")
                os.replace(tmp_path, target)
                tmp_path = None
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
            downloaded.append(target)
    finally:
        if bundle_tmp is not None and bundle_tmp.exists():
            bundle_tmp.unlink()
    return tuple(downloaded)


def resolve_esmc_weights(path: str | Path | None = None, *, required: bool = False) -> Path | None:
    """Resolve a local ESM-C weights path without downloading third-party model files."""

    raw = path or os.environ.get(ESMC_WEIGHTS_ENV)
    if raw is None or str(raw).strip() == "":
        if required:
            raise AssetError(
                "ESM-C weights were not provided. Set BETLAS_ESMC_WEIGHTS or pass an explicit path "
                "after obtaining the weights from the upstream ESM-C distribution."
            )
        return None
    resolved = Path(raw).expanduser()
    if not resolved.exists():
        raise AssetError(
            f"ESM-C weights path does not exist: {resolved}. Betlas does not download or redistribute "
            "third-party ESM-C weights."
        )
    return resolved


__all__ = [
    "ASSET_BASE_URL_ENV",
    "ASSET_DIR_ENV",
    "AssetError",
    "asset_file_report",
    "DEFAULT_ASSET_BASE_URL",
    "ESMC_WEIGHTS_ENV",
    "describe_asset",
    "download_asset",
    "list_assets",
    "resolve_asset_path",
    "resolve_esmc_weights",
    "verify_asset",
]

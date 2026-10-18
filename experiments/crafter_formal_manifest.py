"""Hash-bound manifest checks for frozen Crafter transfer experiments."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "numpy", "crafter"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def verify_formal_manifest(
    config: dict, config_path: str | Path, *, require_current_commit: bool = True,
) -> dict:
    if config.get("formal_result") is not True or config.get("diagnostic_only") is not False:
        raise ValueError("formal manifest verification requires explicit formal configuration")
    if config.get("ready_for_formal_training") is not True:
        raise ValueError("formal configuration is not marked ready for training")
    if config.get("formal_manifest_sealed") is not True:
        raise ValueError("formal configuration does not declare a sealed manifest")
    manifest_path = Path(str(config.get("formal_manifest", "")))
    if not manifest_path.is_file():
        raise ValueError("formal manifest file is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_version = str(config.get("formal_manifest_version", "crafter_gather_spt_transfer_formal_manifest_v1"))
    expected_status = f"sealed_formal_manifest_v{manifest_version.rsplit('_v', 1)[-1]}"
    legacy_v1_manifest = (
        manifest_version == "crafter_gather_spt_transfer_formal_manifest_v1"
        and "manifest_version" not in manifest
    )
    if manifest.get("status") != expected_status or (
        manifest.get("manifest_version") != manifest_version and not legacy_v1_manifest
    ):
        raise ValueError("formal manifest is not sealed")
    if manifest.get("formal_result") is not True:
        raise ValueError("formal manifest is not marked for a formal run")
    declared_digest = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    calculated_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    digest_file = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if declared_digest != calculated_digest or not digest_file.is_file():
        raise ValueError("formal manifest digest or detached checksum is missing/invalid")
    if digest_file.read_text(encoding="ascii").strip() != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("formal manifest detached checksum does not match its contents")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if require_current_commit and manifest.get("git_commit") != current_commit:
        raise ValueError("formal manifest git commit does not match the current checkout")
    current_versions = package_versions()
    if manifest.get("runtime", {}).get("packages") != current_versions:
        raise ValueError("formal manifest package versions differ from the active environment")
    expected_python = str(manifest.get("runtime", {}).get("python", "")).split()[:1]
    if expected_python and expected_python[0].split(".")[:2] != sys.version.split()[0].split(".")[:2]:
        raise ValueError("formal manifest Python major/minor version differs from the active environment")
    missing = []
    mismatched = []
    for raw_path, expected_hash in manifest.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            missing.append(raw_path)
        elif sha256_file(path) != expected_hash:
            mismatched.append(raw_path)
    if missing or mismatched:
        raise ValueError(
            f"formal manifest source hash check failed; missing={missing}, mismatched={mismatched}"
        )
    config_hash = sha256_file(config_path)
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("formal config hash differs from the sealed manifest")
    protocol_path = Path(str(config.get("formal_protocol", "")))
    protocol_hash = sha256_file(protocol_path) if protocol_path.is_file() else None
    if not protocol_hash or manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("formal protocol hash differs from the sealed manifest")
    if manifest.get("replica_seeds") != config.get("replica_seeds"):
        raise ValueError("formal manifest replica seeds differ from the run configuration")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "git_commit": manifest["git_commit"],
        "current_git_commit": current_commit,
        "source_sha256": manifest["source_sha256"],
        "runtime_packages": current_versions,
    }

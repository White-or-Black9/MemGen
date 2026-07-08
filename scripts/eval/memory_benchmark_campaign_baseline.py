"""Freeze the pre-campaign EventQA paper package into a read-only manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "memory-benchmark-campaign-baseline/v1"
EVENTQA_PACKAGE_SCHEMA = "eventqa-final-table-package/v1"
REQUIRED_RELATIVE_PATHS = (
    "paper/draft_v0.md",
    "paper/outline.md",
    "paper/main_table_blueprint.md",
    "research_notes/PAPER_SCOPE.md",
    "outputs/mab/eventqa_final_comparison_package.json",
    "outputs/mab/eventqa_paper_artifact_manifest_sha256.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    repo_root: Path, required_paths: list[Path], accepted_commit: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted_commit": accepted_commit,
        "files": {
            str(path.relative_to(repo_root)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in required_paths
        },
    }


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> None:
    for relative, expected in manifest["files"].items():
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"missing baseline file: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"baseline hash mismatch: {relative}")


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _required_paths(repo_root: Path) -> list[Path]:
    return [repo_root / relative for relative in REQUIRED_RELATIVE_PATHS]


def _eventqa_package_schema(repo_root: Path) -> str:
    package_path = repo_root / "outputs/mab/eventqa_final_comparison_package.json"
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    return str(payload.get("schema_version", "missing"))


def create_baseline_manifest(repo_root: Path, accepted_commit: str) -> dict[str, Any]:
    required_paths = _required_paths(repo_root)
    manifest = build_manifest(
        repo_root=repo_root,
        required_paths=required_paths,
        accepted_commit=accepted_commit,
    )
    validate_manifest(manifest, repo_root)
    manifest["git"] = {
        "head": _git_output(repo_root, "rev-parse", "HEAD"),
        "status_short": _git_output(repo_root, "status", "--short"),
    }
    manifest["eventqa_package"] = {
        "path": "outputs/mab/eventqa_final_comparison_package.json",
        "schema_version": _eventqa_package_schema(repo_root),
        "expected_schema_version": EVENTQA_PACKAGE_SCHEMA,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--accepted-paper-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = create_baseline_manifest(repo_root, args.accepted_paper_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote baseline manifest to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

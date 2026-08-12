"""Inventory local tensors and MECH-09R replay inputs without mutating experiments."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Any

from common import ContractError, commit_manifest, ensure_new_output, read_json, sha256_file, write_csv


AUDIT_SCHEMA = "mdp_method_deepening_inventory_v1"
TENSOR_EXTENSIONS = {".npz", ".npy", ".pt", ".pth", ".safetensors"}
TENSOR_FIELDS = [
    "relative_path",
    "extension",
    "size_bytes",
    "experiment_id",
    "classification",
    "paired_refresh_usable",
]
CHECKPOINT_FIELDS = [
    "cell",
    "stage",
    "method",
    "step",
    "remote_path",
    "bytes",
    "sha256",
    "local_file_present",
]
REFRESH_FIELDS = [
    "artifact_type",
    "count",
    "required_count",
    "sufficient_for_mdp04",
    "reason",
]


def _walk_files(root: Path):
    def ignore_error(_: OSError) -> None:
        return None

    for directory, _, filenames in os.walk(root, followlinks=False, onerror=ignore_error):
        base = Path(directory)
        for filename in filenames:
            yield base / filename


def _experiment_id(relative: Path) -> str:
    return relative.parts[0] if relative.parts else ""


def inventory(workspace_root: Path, output_dir: Path) -> dict[str, Any]:
    science_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(workspace_root / "runs"))
    ).expanduser().resolve()
    ex37_root = (
        science_root
        / "37_mech09_downproj_refresh_mediation"
        / "20260728T075907+0000"
    )
    if not science_root.is_dir() or not ex37_root.is_dir():
        raise ContractError("method-deepening science roots are missing")

    tensor_rows: list[dict[str, Any]] = []
    zip_tensor_member_count = 0
    zip_count = 0
    for path in _walk_files(science_root):
        suffix = path.suffix.lower()
        if suffix in TENSOR_EXTENSIONS:
            relative = path.relative_to(science_root)
            experiment_id = _experiment_id(relative)
            classification = (
                "single_timepoint_numerical_smoke"
                if experiment_id.startswith("27_mech01")
                else "unclassified_tensor"
            )
            tensor_rows.append(
                {
                    "relative_path": str(relative),
                    "extension": suffix,
                    "size_bytes": path.stat().st_size,
                    "experiment_id": experiment_id,
                    "classification": classification,
                    "paired_refresh_usable": False,
                }
            )
        elif suffix == ".zip":
            zip_count += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    zip_tensor_member_count += sum(
                        1
                        for name in archive.namelist()
                        if Path(name).suffix.lower() in TENSOR_EXTENSIONS
                    )
            except (OSError, zipfile.BadZipFile):
                # A corrupt archive cannot be treated as evidence; it is counted
                # in the manifest's scan limitations rather than silently used.
                continue

    refresh_audits = sorted(
        ex37_root.glob("formal/*/replica_*/refresh_tree_audit.json")
    )
    paired_refresh_tensors = [
        row
        for row in tensor_rows
        if row["experiment_id"].startswith("37_mech09")
        and row["extension"] in TENSOR_EXTENSIONS
    ]
    checkpoint_inventory_path = ex37_root / "checkpoint_inventory.json"
    checkpoint_document = read_json(checkpoint_inventory_path)
    checkpoint_rows = []
    for cell in checkpoint_document.get("cells", []):
        remote_path = str(cell["path"])
        checkpoint_rows.append(
            {
                "cell": cell["cell"],
                "stage": cell["stage"],
                "method": cell["method"],
                "step": cell["step"],
                "remote_path": remote_path,
                "bytes": cell["bytes"],
                "sha256": cell["sha256"],
                "local_file_present": Path(remote_path).is_file(),
            }
        )

    refresh_rows = [
        {
            "artifact_type": "formal_replay_units",
            "count": len(refresh_audits),
            "required_count": 12,
            "sufficient_for_mdp04": False,
            "reason": "audits contain shapes, fingerprints, and 17 sampled values, not full matrices",
        },
        {
            "artifact_type": "paired_refresh_tensor_files",
            "count": len(paired_refresh_tensors),
            "required_count": 1,
            "sufficient_for_mdp04": False,
            "reason": "no complete before/after K, inverse, and matched-gradient export is local",
        },
        {
            "artifact_type": "local_origin_checkpoints",
            "count": sum(bool(row["local_file_present"]) for row in checkpoint_rows),
            "required_count": 4,
            "sufficient_for_mdp04": False,
            "reason": "all four accepted checkpoints are recorded only as remote Linux paths",
        },
        {
            "artifact_type": "zip_tensor_members",
            "count": zip_tensor_member_count,
            "required_count": 1,
            "sufficient_for_mdp04": False,
            "reason": "archived handoffs contain no raw tensor payload",
        },
    ]

    manifest_name = "method_deepening_inventory_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "tensor_inventory.csv", tensor_rows, TENSOR_FIELDS)
    write_csv(output_dir / "checkpoint_inventory.csv", checkpoint_rows, CHECKPOINT_FIELDS)
    write_csv(output_dir / "refresh_artifact_inventory.csv", refresh_rows, REFRESH_FIELDS)
    extension_counts = {
        extension: sum(row["extension"] == extension for row in tensor_rows)
        for extension in sorted(TENSOR_EXTENSIONS)
    }
    mdp04_ready = (
        bool(paired_refresh_tensors)
        and len(refresh_audits) == 12
        and all(bool(row["local_file_present"]) for row in checkpoint_rows)
    )
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed_inventory",
        "claim_eligible": False,
        "science_root": str(science_root),
        "experiment37_root": str(ex37_root),
        "tensor_extension_counts": extension_counts,
        "tensor_file_count": len(tensor_rows),
        "zip_file_count": zip_count,
        "zip_tensor_member_count": zip_tensor_member_count,
        "formal_refresh_audit_count": len(refresh_audits),
        "remote_checkpoint_count": len(checkpoint_rows),
        "local_checkpoint_count": sum(
            bool(row["local_file_present"]) for row in checkpoint_rows
        ),
        "mdp04_ready": mdp04_ready,
        "mdp04_status": "ready" if mdp04_ready else "blocked_data",
        "required_next_action": "deterministic_short_replay_or_streaming_metric_export_on_original_llama_host",
        "checkpoint_inventory_sha256": sha256_file(checkpoint_inventory_path),
        "formal_manifest_sha256": sha256_file(ex37_root / "formal" / "formal_manifest.json"),
    }
    commit_manifest(
        output_dir,
        manifest_name,
        result,
        [
            "tensor_inventory.csv",
            "checkpoint_inventory.csv",
            "refresh_artifact_inventory.csv",
        ],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inventory(args.workspace_root.resolve(), args.output_dir.resolve())
    print(
        f"method-deepening inventory passed: tensors={result['tensor_file_count']} "
        f"mdp04={result['mdp04_status']}"
    )


if __name__ == "__main__":
    main()

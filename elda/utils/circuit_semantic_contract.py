from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


INTERFACE_BUCKET_TO_SPEC = {
    "size_bucket": "partition_size",
    "stub_bucket": "boundary_stub_count",
    "pin_bucket": "boundary_pin_count",
    "pin_deficit_bucket": "gate_pin_deficit_scaled",
    "underconnected_bucket": "underconnected_gate_fraction_scaled",
    "synthetic_input_bucket": "synthetic_input_proxy_ratio_scaled",
}


def _normalize_dataset_names(dataset_names) -> list[str]:
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        return [dataset_names]
    return [str(name) for name in dataset_names]


def _require_path(path: Path, errors: list[str], label: str) -> None:
    if not path.exists():
        errors.append(f"missing {label}: {path}")


def _require_nonempty_dir(path: Path, errors: list[str], label: str) -> None:
    if not path.exists():
        errors.append(f"missing {label}: {path}")
        return
    if not any(path.iterdir()):
        errors.append(f"empty {label}: {path}")


def validate_semantic_dataset_root(
    root,
    *,
    dataset_names=None,
    schema_fields: Iterable[str] | None = None,
    gate_type_profile_path=None,
    require_gate_profile: bool = False,
) -> dict:
    root = Path(root)
    dataset_names = _normalize_dataset_names(dataset_names)
    schema_fields = [str(field) for field in (schema_fields or [])]
    errors: list[str] = []
    warnings: list[str] = []

    meta_path = root / "meta.pt"
    _require_path(meta_path, errors, "meta.pt")
    meta = None
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    if meta is not None:
        for key in ["version", "net_id", "boundary_stub_id", "splits"]:
            if key not in meta:
                errors.append(f"meta.pt missing key: {key}")
        if "design_to_split" not in meta.get("splits", {}):
            errors.append("meta.pt missing splits.design_to_split")

    needs_partition = "CIRCUIT_PARTITION" in dataset_names
    needs_skeleton = "CIRCUIT_SKELETON" in dataset_names
    if needs_partition:
        _require_nonempty_dir(root / "partitions" / "graphs", errors, "partition graphs")
    if needs_skeleton:
        _require_nonempty_dir(root / "skeletons" / "graphs", errors, "skeleton graphs")
    if needs_partition or needs_skeleton:
        _require_nonempty_dir(root / "manifests", errors, "manifests")

    if schema_fields:
        if meta is None:
            errors.append("schema_fields requested but meta.pt is unavailable")
        else:
            explicit_specs = {
                str(spec.get("name")): spec
                for spec in list(meta.get("schema_token_specs", []))
                if spec.get("name") is not None
            }
            if any(field in {"all", "*", "__all__"} for field in schema_fields):
                if not explicit_specs:
                    errors.append("schema_fields=all requested but meta.schema_token_specs is empty")
                schema_fields_to_check = sorted(explicit_specs.keys())
            else:
                schema_fields_to_check = list(schema_fields)
            interface_bucket_keys = list(meta.get("interface_bucket_keys", []))
            buckets = meta.get("buckets", {})
            for field in schema_fields_to_check:
                if field in explicit_specs:
                    count = int(explicit_specs[field].get("count", 0))
                    if count <= 0:
                        errors.append(f"schema field explicit spec has non-positive count: {field}")
                    continue
                if field not in interface_bucket_keys:
                    errors.append(f"schema field not declared in meta.interface_bucket_keys: {field}")
                    continue
                spec_name = INTERFACE_BUCKET_TO_SPEC.get(field)
                if spec_name is None:
                    errors.append(f"schema field has no bucket mapping: {field}")
                    continue
                upper_bounds = list(buckets.get(spec_name, {}).get("upper_bounds", []))
                if not upper_bounds:
                    errors.append(f"schema field bucket has no upper_bounds: {field} -> {spec_name}")

    if require_gate_profile or gate_type_profile_path is not None:
        profile_path = Path(gate_type_profile_path) if gate_type_profile_path else root / "gate_type_profile.json"
        _require_path(profile_path, errors, "gate_type_profile.json")
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            for key in ["manifest_count", "total_gates", "used_type_count"]:
                if key not in profile:
                    errors.append(f"gate_type_profile.json missing key: {key}")
            if int(profile.get("manifest_count", 0)) <= 0:
                warnings.append(f"gate_type_profile manifest_count is non-positive: {profile_path}")

    return {
        "root": str(root),
        "dataset_names": dataset_names,
        "schema_fields": schema_fields,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def assert_semantic_dataset_root(**kwargs) -> dict:
    report = validate_semantic_dataset_root(**kwargs)
    if report["errors"]:
        raise RuntimeError(
            "semantic dataset root validation failed:\n- " + "\n- ".join(report["errors"])
        )
    return report

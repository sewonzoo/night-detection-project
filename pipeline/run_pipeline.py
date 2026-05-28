"""
Pipeline skeleton for the night detection evaluation flow.

This file validates common inputs and then generates the final metrics table.
It intentionally does not run RetinexFormer or RT-DETRv2 yet; those model
steps can be connected after each teammate's output format is stable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return load_simple_pipeline_yaml(Path(path))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_simple_pipeline_yaml(path: Path) -> dict[str, Any]:
    """Minimal parser for this project's config.yaml when PyYAML is absent."""
    config: dict[str, Any] = {"conditions": {}, "outputs": {}}
    section = None
    current_condition = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()

        if indent == 0:
            if value:
                config[key] = value
                section = None
            else:
                section = key
                current_condition = None
        elif section == "conditions" and indent == 2:
            current_condition = key
            config["conditions"][current_condition] = {}
        elif section == "conditions" and indent == 4 and current_condition:
            config["conditions"][current_condition][key] = value
        elif section == "outputs" and indent == 2:
            config["outputs"][key] = value

    return config


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_paths(config: dict[str, Any]) -> list[str]:
    root = Path(config.get("project_root", ".")).resolve()
    warnings = []

    annotation = resolve(root, config["annotation"])
    if not annotation.exists():
        warnings.append(f"Missing annotation file: {annotation}")

    for condition, spec in config["conditions"].items():
        for key in ("image_dir", "pred_json", "vis_dir"):
            path = resolve(root, spec[key])
            if not path.exists():
                warnings.append(f"Missing {condition}.{key}: {path}")

    for output_path in config["outputs"].values():
        resolve(root, output_path).parent.mkdir(parents=True, exist_ok=True)

    return warnings


def run_metrics_summary(config_path: str) -> None:
    cmd = [sys.executable, "evaluation/summarize_results.py", "--config", config_path]
    subprocess.run(cmd, check=True)


def run_demo_results(config_path: str) -> None:
    cmd = [sys.executable, "evaluation/build_demo_results.py", "--config", config_path]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline skeleton.")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Pipeline config YAML path")
    parser.add_argument("--skip_summary", action="store_true", help="Only validate paths")
    parser.add_argument("--skip_demo", action="store_true", help="Do not build demo_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    warnings = validate_paths(config)
    if warnings:
        print("[WARN] Pipeline validation found missing optional inputs:")
        for warning in warnings:
            print(f"  - {warning}")
        print("[WARN] Existing prediction JSON files can still be summarized.")

    if not args.skip_summary:
        run_metrics_summary(args.config)
    if not args.skip_demo:
        run_demo_results(args.config)


if __name__ == "__main__":
    main()

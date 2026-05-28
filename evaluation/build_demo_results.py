"""
Build demo_results.json for the web demo.

The output groups original, low-light, and enhanced assets by image_id and adds
simple per-image detection counts for quick UI rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITION_TO_DEMO_KEY = {
    "original": "original",
    "low_light_gamma3.0": "low_light",
    "enhanced_retinexformer": "enhanced",
}

DETECTION_COUNT_KEYS = {
    "original": "original_detections",
    "low_light": "low_light_detections",
    "enhanced": "enhanced_detections",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        import yaml
    except ImportError:
        return load_simple_pipeline_yaml(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
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


def load_predictions(pred_json: str | Path) -> dict[str, list[dict[str, Any]]]:
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    predictions = data["results"] if isinstance(data, dict) and "results" in data else data
    return {item["image_id"]: item.get("detections", []) for item in predictions}


def as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def build_demo_results(config: dict[str, Any]) -> list[dict[str, Any]]:
    condition_predictions = {
        condition: load_predictions(spec["pred_json"])
        for condition, spec in config["conditions"].items()
    }
    image_ids = sorted(set().union(*(preds.keys() for preds in condition_predictions.values())))

    results = []
    for image_id in image_ids:
        item: dict[str, Any] = {"image_id": image_id, "metrics": {}}

        for condition, spec in config["conditions"].items():
            demo_key = CONDITION_TO_DEMO_KEY.get(condition, condition)
            detections = condition_predictions[condition].get(image_id, [])
            item[demo_key] = {
                "image_path": as_posix(Path(spec["image_dir"]) / image_id),
                "detection_path": as_posix(Path(spec["vis_dir"]) / image_id),
            }
            metric_key = DETECTION_COUNT_KEYS.get(demo_key, f"{demo_key}_detections")
            item["metrics"][metric_key] = len(detections)

        results.append(item)

    return results


def write_demo_results(results: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo_results.json from configured predictions.")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Pipeline config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    results = build_demo_results(config)
    write_demo_results(results, config["outputs"]["demo_json"])
    print(f"Saved {len(results)} demo items to {config['outputs']['demo_json']}")


if __name__ == "__main__":
    main()

"""
Build final_metrics.csv and final_metrics.md from configured prediction files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval_coco_map import evaluate_prediction


METRIC_COLUMNS = ["condition", "mAP", "AP50", "AP75", "avg_detections", "avg_confidence"]


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


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def collect_metrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = config["annotation"]
    rows = []
    for condition, spec in config["conditions"].items():
        pred_json = spec["pred_json"]
        metrics = evaluate_prediction(pred_json, annotation)
        row = {"condition": condition}
        row.update(metrics)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt(row[col]) if col != "condition" else row[col] for col in METRIC_COLUMNS})


def write_markdown(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Metrics",
        "",
        "| condition | mAP | AP50 | AP75 | avg_detections | avg_confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {mAP} | {AP50} | {AP75} | {avg_detections} | {avg_confidence} |".format(
                condition=row["condition"],
                mAP=fmt(row["mAP"]),
                AP50=fmt(row["AP50"]),
                AP75=fmt(row["AP75"]),
                avg_detections=fmt(row["avg_detections"]),
                avg_confidence=fmt(row["avg_confidence"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize configured evaluation results.")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Pipeline config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = collect_metrics(config)
    write_csv(rows, config["outputs"]["metrics_csv"])
    write_markdown(rows, config["outputs"]["metrics_md"])
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

"""
Evaluate RT-DETRv2 prediction JSON files against COCO annotations.

The project prediction format stores bboxes as [x1, y1, x2, y2] and image IDs
as filenames. COCOeval expects numeric image_id values, category IDs, and
[x, y, width, height] bboxes, so this module performs that conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


COCO_80_TO_91_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def load_prediction_file(pred_json: str | Path) -> list[dict[str, Any]]:
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported prediction JSON format: {pred_json}")


def image_filename_to_id(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem)


def rtdetr_class_to_coco_category_id(class_id: int) -> int:
    if class_id < 0 or class_id >= len(COCO_80_TO_91_CATEGORY_IDS):
        raise ValueError(f"class_id out of COCO range: {class_id}")
    return COCO_80_TO_91_CATEGORY_IDS[class_id]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def prediction_to_coco_results(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coco_results = []
    for item in predictions:
        image_id = image_filename_to_id(item["image_id"])
        for det in item.get("detections", []):
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": rtdetr_class_to_coco_category_id(int(det["class_id"])),
                    "bbox": xyxy_to_xywh([float(v) for v in det["bbox"]]),
                    "score": float(det["score"]),
                }
            )
    return coco_results


def detection_summary(predictions: list[dict[str, Any]]) -> dict[str, float]:
    per_image_counts = [len(item.get("detections", [])) for item in predictions]
    scores = [
        float(det["score"])
        for item in predictions
        for det in item.get("detections", [])
    ]
    return {
        "avg_detections": mean(per_image_counts) if per_image_counts else 0.0,
        "avg_confidence": mean(scores) if scores else 0.0,
    }


def evaluate_coco(annotation_json: str | Path, coco_results: list[dict[str, Any]]) -> dict[str, float | None]:
    annotation_path = Path(annotation_json)
    if not annotation_path.exists():
        return {"mAP": None, "AP50": None, "AP75": None}
    if not coco_results:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for mAP evaluation. "
            "Install it with: pip install pycocotools"
        ) from exc

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }


def evaluate_prediction(pred_json: str | Path, annotation_json: str | Path) -> dict[str, float | None]:
    predictions = load_prediction_file(pred_json)
    coco_results = prediction_to_coco_results(predictions)
    metrics = evaluate_coco(annotation_json, coco_results)
    metrics.update(detection_summary(predictions))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one RT-DETRv2 prediction JSON with COCO mAP.")
    parser.add_argument("--annotation", required=True, help="COCO annotation JSON path")
    parser.add_argument("--pred_json", required=True, help="Project prediction JSON path")
    parser.add_argument("--output_json", default=None, help="Optional metrics JSON output path")
    args = parser.parse_args()

    metrics = evaluate_prediction(args.pred_json, args.annotation)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""
Detection 시각화 스크립트
================================
이미지 + detection JSON 입력 → bbox/class/score를 그려서 저장
"""

import argparse
import colorsys                              # 색 공간 변환 (HSV ↔ RGB)
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont  # 이미지에 그리기
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 색상 및 폰트
# ---------------------------------------------------------------------------
def get_color(class_id: int):

    # class_id 기반 고정 색상 (golden ratio로 골고루 분포)
    hue = (class_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)  # HSV: 색상/채도/명도
    return (int(r * 255), int(g * 255), int(b * 255))


def get_font(size: int = 16):

    # 가독성 좋은 폰트 시도, 없으면 기본 폰트
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()  # 작고 이상한 폰트 (발표 자료에 적합 X)


# ---------------------------------------------------------------------------
# 핵심 시각화 함수
# ---------------------------------------------------------------------------
def draw_detections(image: Image.Image, detections: list, font=None):
    """이미지 위에 bbox/class/score를 그려서 반환."""
    if font is None:
        font = get_font(16)

    img = image.copy()  # 원본 이미지를 직접 변경하지 않기 위한 복사본 생성
    draw = ImageDraw.Draw(img)  # 이미지에 그림 그릴 어댑터 생성

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det.get("class_id", 0)
        class_name = det["class"]
        score = det["score"]

        color = get_color(class_id)
        label = f"{class_name} {score:.2f}"

        # bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 텍스트 크기 측정
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)  # 텍스트가 들어갈 박스 좌표 반환
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)

        # 레이블 위치 (상단 잘림 방지)
        #   1. 레이블이 박스 위에 위치
        #   2. 박스가 이미지 맨 위에 -> 레이블이 박스 안에 위치
        label_y = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 2
        label_x = x1

        # 배경 + 텍스트 그리기
        #   1줄: 텍스트 배경 사각형. bbox와 같은 색으로. +4는 텍스트 주변 여백
        #   2줄: 흰색 텍스트 그리기. 색깔 배경 위에 흰 글씨
        draw.rectangle(
            [label_x, label_y, label_x + tw + 4, label_y + th + 4],
            fill=color,
        )
        draw.text((label_x + 2, label_y + 2), label, fill="white", font=font)

    return img


def visualize_folder(
    image_dir,
    pred_json,
    output_dir,
    image_ids=None,
    max_images=None,
    quality=85,
):
    # 1) JSON 로드 (meta 있는 형식 / 리스트 형식 모두 지원)
    with open(pred_json) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        predictions = data["results"]
        meta = data.get("meta", {})
    else:
        predictions = data
        meta = {}

    print(f"[INFO] Loaded {len(predictions)} predictions from {pred_json}")
    if meta:
        print(f"       Source: threshold={meta.get('threshold')}, model={meta.get('model')}")

    # 2) 필터링
    if image_ids:
        id_set = set(image_ids)
        predictions = [p for p in predictions if p["image_id"] in id_set]
        print(f"[INFO] Filtered by image_ids: {len(predictions)} predictions")

    if max_images:
        predictions = predictions[:max_images]
        print(f"[INFO] Limited to first {max_images}")

    if not predictions:
        print("[ERROR] Nothing to visualize")
        return

    # 3) 출력 폴더
    os.makedirs(output_dir, exist_ok=True)
    font = get_font(16)

    # 4) 그리기
    success, skipped, failed = 0, 0, []

    for pred in tqdm(predictions, desc="Visualizing"):
        img_name = pred["image_id"]
        img_path = Path(image_dir) / img_name

        if not img_path.exists():
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            vis = draw_detections(image, pred["detections"], font=font)

            save_path = Path(output_dir) / img_name
            if save_path.suffix.lower() in (".jpg", ".jpeg"):
                vis.save(save_path, "JPEG", quality=quality, optimize=True)
            else:
                vis.save(save_path)
            success += 1
        except Exception as e:
            failed.append((img_name, str(e)))

    # 5) 요약
    print("\n" + "=" * 60)
    print(f"[DONE] Output dir: {output_dir}")
    print(f"  Saved          : {success}")
    print(f"  Skipped (no img): {skipped}")
    print(f"  Failed         : {len(failed)}")
    if failed:
        print(f"  Failed sample  : {failed[:3]}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize RT-DETRv2 detections")
    parser.add_argument("--image_dir", required=True,
                        help="원본 이미지 폴더 (원본/야간/복원 중 하나)")
    parser.add_argument("--pred_json", required=True,
                        help="detect_rtdetr.py가 만든 JSON")
    parser.add_argument("--output_dir", required=True,
                        help="시각화 결과 저장 폴더")
    parser.add_argument("--image_ids", nargs="*", default=None,
                        help="특정 image_id만 (공백 구분)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="최대 이미지 수")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG 품질 1~95 (default 85)")
    args = parser.parse_args()

    visualize_folder(
        image_dir=args.image_dir,
        pred_json=args.pred_json,
        output_dir=args.output_dir,
        image_ids=args.image_ids,
        max_images=args.max_images,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()

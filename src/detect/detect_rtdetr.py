"""
RT-DETRv2 객체 탐지 스크립트
================================
이미지 폴더를 입력받아 RT-DETRv2로 추론하고 결과를 JSON으로 저장
"""
import argparse           # 커맨드라인 인자 파싱 (--input_dir 같은 옵션 받기)
import json               # JSON 파일 읽기/쓰기
import os                 # 파일/폴더 경로 다루기
import time               # 추론 소요 시간 측정
from pathlib import Path  # 경로 객체

import torch              # PyTorch (GPU 추론, 텐서 연산)
from PIL import Image     # Python Image Library로 이미지 로드/저장
from tqdm import tqdm     # 진행률 표시 바

# HuggingFace transformers 라이브러리의 "RT-DETRv2 모델 클래스"와 "전처리기" 가져오기
#   RTDetrV2ForObjectDetection: 모델 본체. 입력 텐서 받아 logits(클래스 점수), pred_boxes(박스 좌표) 출력
#   RTDetrImageProcessor: 전처리/후처리 담당. 이미지를 모델이 받을 수 있는 형태로 변환하고, 결과를 사람이 이해할 수 있는 형태로 되돌림
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor


# ---------------------------------------------------------------------------
# Model loading: RT-DETRv2 모델과 프로세서 로드
# ---------------------------------------------------------------------------
def load_model(model_name: str = "PekingU/rtdetr_v2_r50vd", device: str = "cuda"):
    print(f"[INFO] Loading model: {model_name}")

    # HuggingFace Hub에서 사전학습 가중치 다운로드 후 로드
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrV2ForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[INFO] Model loaded on {device}")
    return processor, model


# 경로 리스트를 각각 PIL 이미지로 열고 RGB 변환. 손상 파일은 None 반환
def load_images_batch(paths):
    images, valid_paths = [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p.name}: {e}")
    return images, valid_paths


def detect_folder(
    input_dir: str,
    output_json: str,
    threshold: float = 0.5,
    batch_size: int = 8,
    model_name: str = "PekingU/rtdetr_v2_r50vd",
    max_images: int = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(model_name, device)

    # 모델 출력은 클래스 번호(0~79) -> 사람이 읽기 위한 라벨 꺼내야함
    id2label = model.config.id2label

    # 이미지 파일 수집 (확장자 필터링)
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in exts
    ])

    # 10개로만 우선적으로 테스트 하기 위해
    if max_images:
        image_paths = image_paths[:max_images]

    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print(f"[INFO] Total images to process: {len(image_paths)}")
    print(f"[INFO] Threshold: {threshold}, Batch size: {batch_size}")

    all_results = []
    failed = []
    start_time = time.time()

    # 한 번에 배치 크기만큼 묶어서 추론
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Detecting"):
        batch_paths = image_paths[i:i + batch_size]
        images, valid_paths = load_images_batch(batch_paths)

        if not images:
            failed.extend([p.name for p in batch_paths])
            continue

        # 이미지 -> 모델 입력으로 변환. 좌표계 통일: PIL.size = (W, H) → 모델 입력은 (H, W)
        # 안 하면 bbox 좌표 가로, 세로 바뀜
        sizes = torch.tensor(
            [img.size[::-1] for img in images]
        ).to(device)

        try:
            # PIL 이미지를 모델 입력 형식으로 변환 (리사이즈, 정규화, 텐서화)
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():            # 학습때만 gradient 필요
                outputs = model(**inputs)    # 모델의 raw 출력: raw logits + 박스 좌표(정규화된 0~1 범위)

            # raw 출력을 사람이 읽을 수 있는 형식으로 변환
            #   1. softmax: logits -> score(확률)
            #   2. 정규화된 박스좌표를 원본 이미지 크기로 복원
            #   3. threshold 이하 score 필터링
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=sizes,
                threshold=threshold,
            )
        except Exception as e:
            print(f"[WARN] Batch failed: {e}")
            failed.extend([p.name for p in valid_paths])
            continue

        # 결과 정리
        for path, result in zip(valid_paths, results): # 배치 안의 각 이미지 처리
            detections = []
            for score, label, box in zip(              # 한 이미지에서 탐지된 객체 각각 처리
                result["scores"], result["labels"], result["boxes"]
            ):
                detections.append({
                    "class": id2label[label.item()],
                    "class_id": int(label.item()),
                    "score": round(float(score.item()), 4),
                    "bbox": [round(float(x), 2) for x in box.tolist()],
                })
            all_results.append({
                "image_id": path.name,
                "detections": detections,
            })

        # 최종 JSON 형식
        # {
        #   "image_id": "000000000139.jpg",
        #   "detections": [
        #     {
        #       "class": "person",
        #       "class_id": 0,
        #       "score": 0.9123,
        #       "bbox": [120.45, 80.21, 240.11, 380.07]
        #     }
        #   ]
        # }

    elapsed = time.time() - start_time

    # 메타 정보 + 결과 저장
    output_data = {
        "meta": {
            "model": model_name,
            "input_dir": str(input_dir),
            "threshold": threshold,
            "batch_size": batch_size,
            "total_images": len(image_paths),
            "successful": len(all_results),
            "failed": len(failed),
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": all_results,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    # 요약 출력
    print("\n" + "=" * 60)
    print(f"[DONE] Saved to: {output_json}")
    print(f"  Total images   : {len(image_paths)}")
    print(f"  Successful     : {len(all_results)}")
    print(f"  Failed         : {len(failed)}")
    print(f"  Elapsed        : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if all_results:
        total_dets = sum(len(r["detections"]) for r in all_results)
        avg_dets = total_dets / len(all_results)
        print(f"  Total detections: {total_dets}")
        print(f"  Avg per image  : {avg_dets:.2f}")
    if failed:
        print(f"  Failed files   : {failed[:5]}{'...' if len(failed) > 5 else ''}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI: 명령어로 인자 받기 위함 -> 다른 데이터 돌릴 때도 재사용 가능
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RT-DETRv2 detection on image folder")
    parser.add_argument("--input_dir", required=True, help="Input image folder")
    parser.add_argument("--output_json", required=True, help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8, lower if OOM)")
    parser.add_argument("--model_name", default="PekingU/rtdetr_v2_r50vd",
                        help="HuggingFace model name")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for testing)")
    args = parser.parse_args()

    detect_folder(
        input_dir=args.input_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_name=args.model_name,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()

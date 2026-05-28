# Evaluation Summary

## 1. 평가 데이터 수

현재 평가 단위는 COCO val2017에서 추출한 `sample_200` 기준 200장이다.

현재 repository에는 `results/detection/pred_*.json`과 시각화 결과 200장씩은 포함되어 있지만, `data/sample_200/original/`, `data/sample_200/low_light_gamma3.0/`, `data/sample_200/annotations_sample_200.json`은 포함되어 있지 않다. 따라서 현재 산출 가능한 값은 평균 탐지 개수와 평균 confidence이며, COCO mAP 계열 지표는 annotation 파일이 들어오면 재계산된다.

## 2. 평가 조건

| condition | 설명 | prediction |
|---|---|---|
| `original` | 원본 COCO 이미지에 RT-DETRv2 적용 | `results/detection/pred_original.json` |
| `low_light_gamma3.0` | 감마 보정 `gamma=3.0` 야간 합성 이미지에 RT-DETRv2 적용 | `results/detection/pred_low_light.json` |
| `enhanced_retinexformer` | 야간 합성 이미지를 RetinexFormer로 복원한 뒤 RT-DETRv2 적용 | `results/detection/pred_enhanced.json` |

## 3. 사용 Annotation

표준 annotation 경로는 다음으로 고정한다.

```text
data/sample_200/annotations_sample_200.json
```

이 파일은 COCO 형식이어야 하며, `images`, `annotations`, `categories`를 포함해야 한다.

## 4. mAP 계산 방식

`evaluation/eval_coco_map.py`는 프로젝트 prediction JSON을 COCO detection 결과 형식으로 변환한 뒤 `pycocotools.COCOeval`의 bbox 평가를 사용한다.

변환 규칙:

```text
image_id: "000000000139.jpg" -> 139
bbox: [x1, y1, x2, y2] -> [x, y, width, height]
class_id: RT-DETR/HuggingFace COCO 80-class index -> COCO category_id
```

산출 지표:

```text
mAP  = COCO AP@[0.50:0.95]
AP50 = COCO AP@0.50
AP75 = COCO AP@0.75
avg_detections = 이미지당 평균 탐지 수
avg_confidence = 전체 detection confidence 평균
```

## 5. Original 결과

현재 prediction 기준:

```text
avg_detections = 7.4700
avg_confidence = 0.7635
```

mAP/AP50/AP75는 `annotations_sample_200.json` 추가 후 계산한다.

## 6. Low-light 결과

현재 prediction 기준:

```text
avg_detections = 5.9100
avg_confidence = 0.7384
```

원본 대비 평균 탐지 개수가 감소했다.

## 7. Enhanced 결과

현재 prediction 기준:

```text
avg_detections = 6.0000
avg_confidence = 0.7278
```

야간 합성 이미지 대비 평균 탐지 개수는 소폭 증가했지만, mAP 기준의 최종 개선 여부는 annotation 기반 평가로 확인해야 한다.

## 8. 최종 비교표

현재 자동 생성 결과:

| condition | mAP | AP50 | AP75 | avg_detections | avg_confidence |
|---|---:|---:|---:|---:|---:|
| original | NA | NA | NA | 7.4700 | 0.7635 |
| low_light_gamma3.0 | NA | NA | NA | 5.9100 | 0.7384 |
| enhanced_retinexformer | NA | NA | NA | 6.0000 | 0.7278 |

최신 표는 항상 아래 파일을 기준으로 한다.

```text
results/metrics/final_metrics.csv
results/metrics/final_metrics.md
```

## 9. 발표용 해석 문장

원본 이미지 대비 감마 보정 야간 이미지에서는 평균 탐지 개수가 감소하는 경향을 보였다. RetinexFormer 복원 이미지를 적용한 경우 평균 탐지 개수는 야간 합성 이미지보다 소폭 회복되었지만, 전체 탐지 정확도 개선 여부는 COCO annotation을 이용한 mAP/AP50 평가로 최종 확인해야 한다. 이는 저조도 복원이 객체 탐지 성능 개선에 기여할 가능성이 있으나, 복원 결과가 탐지 모델의 bbox/class 정확도와 얼마나 잘 정합되는지가 중요함을 시사한다.

# Baseline Summary

**담당자:** 주세원  
**작성일:** 2026-05-26  
**상태:** 초안 (mAP 수치 업데이트 예정 — 김태익, 정시원 협업)

---

## 1. 사용 데이터셋

| 항목 | 내용 |
|---|---|
| 데이터셋 | COCO val2017 |
| 전체 이미지 수 | 5,000장 |
| 실험 샘플 수 | 200장 (`data/sample_200/`) |
| Annotation | `instances_val2017.json` → `annotations_sample_200.json` 필터링 |
| 카테고리 수 | 80개 |

---

## 2. 야간 합성 방법 (감마 보정)

감마 보정(Gamma Correction)을 이용해 원본 COCO 이미지를 야간 환경처럼 어둡게 변환했다.

```
I_low(x) = (I(x) / 255) ^ γ × 255

I(x)     : 원본 이미지 픽셀값 (0~255)
γ        : 감마값 (클수록 더 어두워짐)
I_low(x) : 야간 합성 이미지 픽셀값
```

| 항목 | 값 |
|---|---|
| 사용 감마값 (γ) | 3.0 |
| 저장 경로 | `data/synthetic/coco_gamma3.0/` |
| 샘플 경로 | `data/sample_200/low_light_gamma3.0/` |

---

## 3. 밝기 분석 결과

| 조건 | 평균 픽셀 밝기 | 밝기 감소율 |
|---|---|---|
| Original | `[측정값]` | — |
| Low-light (γ=3.0) | `[측정값]` | `[측정값]`% |

> 📌 `results/analysis/brightness_histogram.png` 참고

**해석:**  
감마 보정(γ=3.0) 적용 시, 이미지의 전반적인 밝기 분포가 낮은 영역으로 이동한다.  
이는 물체의 경계선과 질감 정보를 약화시켜, 탐지 모델의 성능 하락을 유발할 수 있다.

---

## 4. 실험 조건

| 조건 | 설명 |
|---|---|
| **Baseline (A)** | Original COCO image → RT-DETRv2 |
| **성능 하락 확인 (B)** | Low-light image (γ=3.0) → RT-DETRv2 |
| **제안 방법 (C)** | Low-light image → RetinexFormer → RT-DETRv2 |

> 조건 C는 강희범, 김태익, 정시원 협업으로 추후 측정

---

## 5. 탐지 결과 (mAP)

> **⚠️ 수치 업데이트 필요** — 김태익(탐지 실행), 정시원(mAP 계산) 완료 후 채울 것

| 조건 | mAP@0.5 | mAP@0.5:0.95 | 비고 |
|---|---|---|---|
| A: Original | 68.43% | 53.37% | Baseline |
| B: Low-light (γ=3.0) | 53.79% | 40.81% | 성능 하락 확인용 |
| C: RetinexFormer 복원 후 | 49.84% | 36.90% | 제안 방법 (추후 측정) |

---

## 6. 성능 하락 요약 (채울 예정)

```
[작성 예시]
원본 대비 야간 합성 이미지에서 mAP@0.5가 XX.X → XX.X로 XX.X% 하락했다.
이는 감마 보정으로 인한 밝기 감소가 RT-DETRv2의 특징 추출을 방해했기 때문으로 해석된다.
```

> 수치 확보 후 위 양식에 맞게 작성

---

## 7. 발표용 해석 문장 (초안)

> "저희는 COCO val2017 데이터셋에서 200장을 샘플링하여 감마 보정(γ=3.0)으로 야간 이미지를 합성했습니다.  
> 원본 대비 밝기가 크게 감소했으며, 이로 인해 RT-DETRv2의 탐지 성능이 mAP 기준 약 XX% 하락하는 것을 확인했습니다.  
> 이 결과는 야간 환경에서의 이미지 복원 모듈의 필요성을 실험적으로 뒷받침합니다."

---

## 8. 산출물 경로

| 파일 | 경로 |
|---|---|
| 노트북 | `notebooks/01_data_pipeline.ipynb` |
| 분석 노트북 | `notebooks/02_sample_and_analysis.ipynb` |
| 파이프라인 문서 | `docs/research/data_pipeline.md` |
| Baseline 요약 | `docs/research/baseline_summary.md` (이 파일) |
| 밝기 히스토그램 | `results/analysis/brightness_histogram.png` |
| 감마 예시 이미지 | `results/analysis/gamma_comparison_samples.png` |
| sample_200 | `data/sample_200/` |

---

## 9. 주세원 체크리스트

- [x] GitHub Repository 생성
- [x] COCO val2017 다운로드
- [x] 감마 보정 야간 이미지 생성 (5,000장)
- [x] 데이터 파이프라인 노트북 작성
- [x] sample_200 생성
- [x] original / synthetic 파일명 매칭 확인
- [x] brightness histogram 생성
- [x] baseline_summary.md 초안 작성
- [ ] mAP 수치 채우기 (김태익, 정시원 결과 대기)
- [ ] 발표용 해석 문장 완성

# Data Pipeline 사용법

**담당자:** 주세원  
**작성일:** 2026-05-24  
**노트북:** `notebooks/01_data_pipeline.ipynb`

---

## 개요

COCO val2017 이미지를 감마 보정(Gamma Correction)으로 야간처럼 어둡게 변환하는 파이프라인입니다.  
생성된 야간 합성 이미지는 RetinexFormer(복원) → RT-DETRv2(탐지) 파이프라인의 입력으로 사용됩니다.

---

## 1. 사전 준비

### Google Colab에서 Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')
```

마운트 후 아래 경로가 기준이 됩니다:

```
/content/drive/MyDrive/night-detection-project/
```

### 필요한 라이브러리

Colab 기본 환경에서 추가 설치 없이 사용 가능합니다.

```python
import os, json, shutil, random
import cv2
import numpy as np
from PIL import Image
```

---

## 2. COCO 데이터 다운로드

```bash
# Colab 셀에서 실행
!wget http://images.cocodataset.org/zips/val2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!unzip -q val2017.zip -d /content/coco/images/
!unzip -q annotations_trainval2017.zip -d /content/coco/
```

다운로드 후 Drive로 복사:

```python
import shutil
shutil.copytree('/content/coco', '/content/drive/MyDrive/night-detection-project/data/raw/coco')
```

> ⚠️ 전체 다운로드 및 복사는 약 20~30분 소요됩니다.

---

## 3. 감마 보정 실행 방법

```python
import cv2
import numpy as np
import os

ORIG_DIR  = '/content/drive/MyDrive/night-detection-project/data/raw/coco/images/val2017'
SYNTH_DIR = '/content/drive/MyDrive/night-detection-project/data/synthetic/coco_gamma3.0'
os.makedirs(SYNTH_DIR, exist_ok=True)

gamma = 3.0

for fname in os.listdir(ORIG_DIR):
    img = cv2.imread(os.path.join(ORIG_DIR, fname))
    if img is None:
        continue
    img_norm  = img / 255.0
    img_dark  = (img_norm ** gamma) * 255.0
    img_dark  = img_dark.astype(np.uint8)
    cv2.imwrite(os.path.join(SYNTH_DIR, fname), img_dark)
```

**감마값 의미:**

| γ 값 | 어두운 정도 |
|---|---|
| 2.0 | 약간 어두움 |
| 3.0 | 야간 수준 (기본값) |
| 5.0 | 매우 어두움 |

---

## 4. 생성되는 폴더 구조

```
night-detection-project/
├── data/
│   ├── raw/
│   │   └── coco/
│   │       ├── images/
│   │       │   └── val2017/          ← 원본 이미지 5,000장
│   │       └── annotations/
│   │           └── instances_val2017.json
│   ├── synthetic/
│   │   └── coco_gamma3.0/            ← 야간 합성 이미지 5,000장
│   └── sample_200/
│       ├── original/                 ← 실험용 원본 200장
│       ├── low_light_gamma3.0/       ← 실험용 야간 200장
│       └── annotations_sample_200.json
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   └── 02_sample_and_analysis.ipynb
├── results/
│   └── analysis/
│       ├── brightness_histogram.png
│       └── gamma_comparison_samples.png
└── docs/
    └── research/
        ├── data_pipeline.md          ← 이 파일
        └── baseline_summary.md
```

---

## 5. 경로 정리

| 항목 | 경로 |
|---|---|
| 원본 이미지 | `data/raw/coco/images/val2017/` |
| 야간 합성 이미지 | `data/synthetic/coco_gamma3.0/` |
| Annotation 파일 | `data/raw/coco/annotations/instances_val2017.json` |
| 실험용 샘플 원본 | `data/sample_200/original/` |
| 실험용 샘플 야간 | `data/sample_200/low_light_gamma3.0/` |
| 샘플 Annotation | `data/sample_200/annotations_sample_200.json` |

---

## 6. 예상 실행 시간

| 작업 | 소요 시간 |
|---|---|
| COCO val2017 다운로드 | 약 10~15분 |
| Drive 복사 | 약 10~20분 |
| 감마 보정 5,000장 생성 | 약 10~15분 |
| sample_200 분리 | 약 1~2분 |
| Brightness histogram 생성 | 약 1~2분 |

> 총 약 30~50분 소요 (Colab 기준)

---

## 7. 실행 확인 코드

```python
# 최종 확인
BASE = '/content/drive/MyDrive/night-detection-project'

paths = {
    '원본 이미지':      f'{BASE}/data/raw/coco/images/val2017',
    '야간 합성 이미지':  f'{BASE}/data/synthetic/coco_gamma3.0',
    'sample 원본':      f'{BASE}/data/sample_200/original',
    'sample 야간':      f'{BASE}/data/sample_200/low_light_gamma3.0',
}

for name, path in paths.items():
    n = len(os.listdir(path)) if os.path.isdir(path) else 0
    print(f'{name}: {n}장')
```

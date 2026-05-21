# 야간 이미지 객체 탐지 성능 개선

## 프로젝트 개요

야간 환경에서 촬영된 이미지의 객체 탐지 성능을 개선하는 프로젝트입니다.  
저조도 이미지 향상(Low-light Enhancement) 모델과 객체 탐지(Object Detection) 모델을 결합하여 야간 탐지 정확도를 높이는 것을 목표로 합니다.

## 사용 모델

| 역할 | 모델 |
|------|------|
| 저조도 이미지 향상 | [RetinexFormer](https://github.com/caiyuanhao1998/RetinexFormer) |
| 객체 탐지 | [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) |

## 파이프라인

```
야간 입력 이미지 → RetinexFormer (이미지 향상) → RT-DETRv2 (객체 탐지) → 결과
```

## 폴더 구조

```
night-detection-project/
├── data/
│   ├── raw/          # 원본 야간 이미지 데이터셋
│   └── synthetic/    # 합성 야간 데이터 (증강 등)
├── notebooks/        # 실험 및 분석용 Jupyter 노트북
├── src/
│   ├── enhance/      # RetinexFormer 관련 코드
│   └── detect/       # RT-DETRv2 관련 코드
├── docs/
│   ├── meetings/     # 회의록
│   └── research/     # 논문 정리 및 참고자료
└── results/          # 실험 결과 (이미지, 로그, 지표)
```

## 팀원

- (팀원 목록 작성)

## 참고 논문

- RetinexFormer: *RetinexFormer: One-stage Retinex-based Transformer for Low-light Image Enhancement* (ICCV 2023)
- RT-DETRv2: *RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer* (2024)

# 1. 사용 모델: RetinexFormer
본 실험에서는 저조도 이미지 개선을 위해 RetinexFormer를 사용했습니다.
RetinexFormer는 Retinex 이론 기반의 One-stage Transformer 모델로, 기존 모델들이 조도 복원과 노이즈 제거를 별도 단계로 처리하던 것을 하나의 파이프라인으로 통합한 것이 핵심입니다.
 
## ORF (One-stage Retinex-based Framework)
기존 Retinex 이론은 입력 이미지가 노이즈 없이 깨끗하다고 가정합니다.

I = R ⊙ L

그러나 실제 저조도 이미지에는 노이즈와 색상 왜곡이 함께 존재합니다.
ORF는 반사율과 조도 각각에 perturbation term을 도입해 이를 반영합니다.

I = (R + R̂) ⊙ (L + L̂)

이후 light-up map L̄ 을 추정해 이미지를 밝히는 동시에 노이즈와 색상 왜곡을 한 번에 제거합니다.
조도를 밝혀도 노이즈가 함께 증폭되지 않는 것이 핵심입니다.
 
## IGT (Illumination-Guided Transformer)
CNN의 한계인 좁은 수용 영역을 극복하기 위해 Transformer의 전역적 self-attention을 활용합니다.
ORF가 추출한 조도 정보를 self-attention 연산에 직접 주입해, 어두운 영역과 밝은 영역을 구분하여 각각에 맞는 복원을 수행합니다.

13개 벤치마크에서 SOTA를 달성했으며, 기존 최고 성능 모델인 SNR-Net 대비 파라미터 40%, 연산량 59% 수준으로 경량화되어 있습니다.

## perturbation term(추가설명)

perturbation term는, 기존 Retinex 이론의 한계를 보완하기 위해 도입된 개념입니다.
기존 Retinex 이론은 이렇게 가정합니다.

I = R ⊙ L

입력 이미지(I)가 반사율(R)과 조도(L)로 깔끔하게 분리된다는 건데, 이 수식은 이미지에 노이즈나 색상 왜곡이 없다고 가정합니다. 실제 저조도 이미지에는 맞지 않는 가정입니다.
그래서 RetinexFormer는 여기에 perturbation term R̂ 과 L̂ 을 추가합니다.

I = (R + R̂) ⊙ (L + L̂)

R̂ : 반사율에 숨어있는 노이즈와 아티팩트(객체의 형태)
L̂ : 조도 추정 오차로 인한 색상 왜곡과 과노출/저노출(객체의 색상)

# 2. 입력 데이터: COCO gamma=3.0 synthetic image
Retinexformer의 성능을 정밀하게 평가하기 위해 합성 저조도 데이터셋을 구축하여 입력 데이터로 사용했습니다.
 
### 베이스 데이터셋
컴퓨터 비전 분야의 표준 벤치마크인 COCO val2017 데이터셋에서 무작위로 추출한 **200장의 이미지(sample_200)**를 기반으로 합니다.
 
### 합성 야간 환경 구축 (γ=3.0)
원본 이미지에 감마 보정(Gamma Correction) γ=3.0 을 적용하여 픽셀 값을 강제로 감쇠시켰습니다.
이를 통해 가로등만 켜진 야간 도로변이나 광량이 극도로 제한된 실내 야간 환경을 인공적으로 모사했습니다.
 
### 데이터 일관성
원본(Original)과 야간 합성본(Low-light)은 1:1 매칭 구조로 구성되어 있습니다.
저조도 환경에서 손실된 시각 정보가 Retinexformer를 통해 픽셀 단위로 얼마나 정확하게 복원되는지 정밀하게 비교 분석할 수 있습니다.

# 3. 출력 데이터 경로


```text
data/
├── enhanced/retinexformer_gamma3.0/      # Retinexformer 복원 이미지 200장
└── results/enhancement/
                ├── success_cases/           # 복원 성공 사례 이미지 5장
                └── failure_cases/           # 복원 실패 사례 이미지 5장
```

# 4. 복원 이미지 전후 예

<img width="1668" height="442" alt="image" src="https://github.com/user-attachments/assets/ee5413ca-a7ae-43d5-99a3-90f498b1b8b1" />
<img width="1663" height="478" alt="image" src="https://github.com/user-attachments/assets/3233bd55-431a-4569-92cc-ab43fcee7e1e" />

# 5. 복원 이미지 성공 사례

<img width="1013" height="576" alt="image" src="https://github.com/user-attachments/assets/cccd76e0-24f2-4fb3-9eed-5a99b1357a23" />
주방 환경 이미지로, 저조도에서 복원이 잘 된 사례입니다.
복원 전 (Synthetic Night)

냉장고, 가스레인지, 수납장 등 주방 구조물이 거의 식별 불가
전체적으로 암흑에 가까운 수준으로 디테일 손실이 심함

복원 후 (Enhanced Result)

냉장고 문 손잡이, 수납장 색상, 바닥 타일 패턴까지 선명하게 복원
색상 왜곡 없이 자연스러운 톤 유지
구조물 간 경계선이 명확하게 살아남

# 6. 복원 이미지 실패 사례

<img width="1100" height="568" alt="image" src="https://github.com/user-attachments/assets/72b187b0-2d58-47b8-9bae-ded68b159362" />
야외 역광 환경의 STOP 표지판 이미지로, 복원이 실패한 사례입니다.
복원 전 (Synthetic Night)

전체적으로 어둡지만 STOP 표지판 형태는 식별 가능
하늘 영역이 상대적으로 밝아 역광 구조가 존재

복원 후 (Enhanced Result)

하늘과 표지판 경계 부근에 붉은 색상 왜곡이 심하게 발생
자연스럽지 않은 노이즈 패턴이 이미지 상단에 집중

# 7. 한계 (Limitations)
본 실험을 통해 도출된 RetinexFormer의 기술적 한계점은 다음과 같습니다.
 
### 1. 원천 정보 소실
저조도 수준이 극단적으로 심해 윤곽 정보가 완전히 유실된 경우, 복원 모델은 전체적인 밝기만 균일하게 올릴 뿐 소실된 사물의 형태 자체를 재구축하지 못합니다.
 
### 2. 왜곡 발생
어두운 영역을 강제로 밝히는 과정에서 미세한 색상 불균형과 픽셀 왜곡(Artifact)이 발생합니다. 이로 인해 사물의 본래 질감이 손상되거나 시각적 부자연스러움이 유발되는 한계가 있습니다.
 
### 3. 소형 객체 및 밀집 영역에서의 에지 보존력 부족
객체가 밀집된 장면이나 픽셀 크기가 작은 소형 객체 영역에서는 Transformer의 어텐션이 공간적 경계를 명확히 구분하지 못하고 뭉개는 경향이 있습니다. 이로 인해 미세 영역 복원 정확도가 상대적으로 낮습니다.

# 8. 발표용 해석 문장
RetinexFormer가 실제 야간 환경에서 객체 탐지 성능 향상에 기여할 수 있는지를 검증해보았습니다.
<br>

RetinexFormer는 기존 Retinex 이론에 perturbation term을 도입해 조도 복원과 노이즈 제거를 하나의 파이프라인으로 처리합니다. ORF가 light-up map을 추정해 이미지를 밝히는 동시에, IGT가 조도 정보를 Self-Attention에 주입해 어두운 영역과 밝은 영역을 구분하여 복원합니다. 이 구조 덕분에 단순히 밝기만 올리는 기존 방식과 달리 노이즈 증폭 없이 자연스러운 복원이 가능합니다.
<br>

실험 결과, 조도 분포가 균일하고 구조가 단순한 실내 환경에서는 원본에 가까운 수준의 복원이 가능했습니다. 그러나 역광이나 극단적인 명암 대비가 존재하는 야외 환경에서는 IGT의 조도 추정이 혼란을 일으켜 색상 왜곡과 아티팩트가 발생하는 한계를 확인했습니다. 이는 조도 분포가 복잡한 실제 야간 환경에 바로 적용되기 위해서는 추가적인 도메인 적응이나 후처리 과정이 필요함을 시사합니다.
<br>

의의는 단순한 복원 성능 측정을 넘어, 저조도 이미지 개선 모델이 야간 객체 탐지 파이프라인의 전처리 단계로서 실질적으로 기여할 수 있는 조건과 한계를 동시에 확인하였습니다.

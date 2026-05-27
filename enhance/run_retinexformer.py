from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 핵심 추가: 필수 라이브러리 및 모델 소스코드가 없으면 자동 다운로드
!pip install lmdb pyyaml addict

BASE_DIR = '/content/drive/MyDrive/night-detection-project'
RETINEXFORMER_PATH = os.path.join(BASE_DIR, 'Retinexformer')

if not os.path.exists(RETINEXFORMER_PATH):
    print("팀원 드라이브에 모델 코드가 없어 GitHub에서 자동으로 다운로드합니다...")
    # 프로젝트 최상위 폴더가 없으면 미리 생성
    os.makedirs(BASE_DIR, exist_ok=True)
    !git clone https://github.com/caiyuanhao1998/Retinexformer.git $RETINEXFORMER_PATH
else:
    print("✅ 이미 Retinexformer 모델 코드가 드라이브에 존재합니다.")

BASE_DIR = '/content/drive/MyDrive/night-detection-project'

# [입력 데이터셋 경로]
SAMPLE_200_DIR = os.path.join(BASE_DIR, 'data/sample_200')
RAW_DIR = os.path.join(SAMPLE_200_DIR, 'original')
SYNTH_DIR = os.path.join(SAMPLE_200_DIR, 'low_light_gamma3.0')

# [출력 데이터셋 경로] - 요청하신 새 결과물 폴더명 반영
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/enhanced_sample_200')

# 안전장치: 결과물이 저장될 폴더가 없으면 자동으로 생성합니다.
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅ 입력 야간 이미지 경로: {SYNTH_DIR}")
print(f"✅ 최종 복원 결과 저장소: {OUTPUT_DIR}")

RETINEXFORMER_PATH = os.path.join(BASE_DIR, 'Retinexformer')
sys.path.insert(0, RETINEXFORMER_PATH)

try:
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 사용 디바이스: {device} (속도를 위해 가급적 GPU/Cuda 환경을 권장합니다.)")

    # 모델 초기화 (in_channels=3, out_channels=3)
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]).to(device)

    # 가중치 파일 경로 연결
    weight_path = os.path.join(RETINEXFORMER_PATH, 'pretrained_weights/LOL_v2_real.pth')
    checkpoint = torch.load(weight_path, map_location=device)

    # 가중치 주입
    model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    model.eval()
    print("✅ RetinexFormer 모델 로드 완료!")
except Exception as e:
    print(f"모델 로드 실패: {e}\n(BASE_DIR 하위에 Retinexformer 폴더와 가중치 파일이 정상 배치되었는지 확인하세요.)")

if os.path.exists(SYNTH_DIR):
    all_files = sorted([f for f in os.listdir(SYNTH_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    target_files = all_files[:200] # 칼같이 200장 타겟팅

    print(f"🚀 총 {len(target_files)}장의 야간 합성 이미지 복원을 시작합니다...")
    transform = transforms.ToTensor()

    with torch.no_grad():
        for filename in tqdm(target_files, desc="이미지 고화질 복원 중"):
            synth_path = os.path.join(SYNTH_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            # 이미 복원된 파일이 목적지 폴더에 존재하면 빠르게 패스 (이어하기 기능)
            if os.path.exists(output_path):
                continue

            try:
                # 이미지 로드 및 텐서 변환
                img = Image.open(synth_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # 해상도 홀수/짝수 에러 방지를 위한 패딩(Padding) 처리
                factor = 16
                h, w = img_tensor.shape[2], img_tensor.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                img_tensor_padded = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')

                # 모델 추론(Inference)
                output_tensor_padded = model(img_tensor_padded)

                # 크기 원복 후 저장
                output_tensor = output_tensor_padded[:, :, :h, :w]
                save_image(output_tensor, output_path)

            except Exception as e:
                print(f"\n파일 복원 에러 발생 ({filename}): {e}")
                continue

    print(f"\n200장 복원 프로세스가 성공적으로 완료되었습니다! 저장 폴더: {OUTPUT_DIR}")
else:
    print(f"에러: 입력 폴더를 찾을 수 없습니다: {SYNTH_DIR}")

VISUALIZE_COUNT = 20

if not os.path.exists(OUTPUT_DIR):
    print("아직 복원된 이미지 결과물이 존재하지 않습니다.")
else:
    # 성공적으로 생성된 결과물 리스트 추출
    generated_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    visualize_files = generated_files[:VISUALIZE_COUNT]

    print(f"총 {len(generated_files)}장의 복원본 중 {len(visualize_files)}장에 대해 시각화를 생성합니다.")

    # 세로로 20장이 겹치지 않게 길쭉한 캔버스 생성
    plt.figure(figsize=(18, 6 * len(visualize_files)))

    for idx, sample_filename in enumerate(visualize_files):
        raw_path = os.path.join(RAW_DIR, sample_filename)
        synth_path = os.path.join(SYNTH_DIR, sample_filename)
        enhanced_path = os.path.join(OUTPUT_DIR, sample_filename)

        # 1열: 낮 원본 이미지 (Original Day GT)
        plt.subplot(len(visualize_files), 3, idx * 3 + 1)
        if os.path.exists(raw_path):
            plt.imshow(Image.open(raw_path))
            plt.title(f"1. Original GT (Day)\n{sample_filename}", fontsize=12)
        else:
            plt.text(0.5, 0.5, "Image Not Found", ha='center', va='center', color='red', fontsize=14)
            plt.title("1. Original GT - Missing", fontsize=12)
        plt.axis('off')

        # 2열: 감마 보정 야간 이미지 (Synthetic Night)
        plt.subplot(len(visualize_files), 3, idx * 3 + 2)
        if os.path.exists(synth_path):
            plt.imshow(Image.open(synth_path))
            plt.title("2. Synthetic Night (Gamma 3.0)", fontsize=12)
        else:
            plt.text(0.5, 0.5, "Image Not Found", ha='center', va='center', color='red', fontsize=14)
            plt.title("2. Synthetic Night - Missing", fontsize=12)
        plt.axis('off')

        # 3열: 최종 새 폴더에 저장 완료된 복원본 (Enhanced Result)
        plt.subplot(len(visualize_files), 3, idx * 3 + 3)
        if os.path.exists(enhanced_path):
            plt.imshow(Image.open(enhanced_path))
            plt.title("3. Enhanced Result (RetinexFormer)", fontsize=12)
        else:
            plt.text(0.5, 0.5, "Not Found", ha='center', va='center', color='orange', fontsize=14)
            plt.title("3. Enhanced - Missing", fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

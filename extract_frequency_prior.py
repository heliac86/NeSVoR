import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
# 사용자 입력란 (경로를 알맞게 채워주세요)
CSV_PATH = "train.csv"
DATA_ROOT = "/data/BraTS2020_TrainingData"
OUTPUT_PATH = "brats_flair_freq_template_16x16.npy"

MODALITY = "flair"
PATCH_SIZE = 16
SAMPLES_PER_PATIENT = 2000  # 환자 한 명당 추출할 랜덤 패치 개수

def extract_frequency_template():
    # 1. CSV 파일에서 훈련셋 환자 ID 목록 불러오기
    print(f"[{CSV_PATH}] 파일에서 훈련셋 목록을 읽어옵니다...")
    df = pd.read_csv(CSV_PATH)
    train_ids = df['Brats20ID'].tolist()
    print(f"총 {len(train_ids)}명의 훈련용 환자 데이터를 찾았습니다.")

    total_amplitude_spectrum = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    valid_patch_count = 0

    # 2. 각 환자별로 순회하며 패치 추출 및 FFT 계산
    for patient_id in tqdm(train_ids, desc="Processing Patients"):
        file_path = os.path.join(DATA_ROOT, patient_id, f"{patient_id}_{MODALITY}.nii")
        
        # .nii.gz 일 경우를 대비한 예외 처리
        if not os.path.exists(file_path):
            file_path += ".gz"
            if not os.path.exists(file_path):
                print(f"경고: {patient_id}의 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue

        # NIfTI 볼륨 로드 (3D 데이터)
        img = nib.load(file_path)
        data = img.get_fdata()

        # BraTS 데이터의 형태는 보통 (240, 240, 155) 입니다. 
        # Z축(Axial 평면)을 기준으로 X-Y 단면을 바라봅니다.
        h, w, d = data.shape

        patient_patches = 0
        attempts = 0
        max_attempts = SAMPLES_PER_PATIENT * 10 # 무한 루프 방지

        while patient_patches < SAMPLES_PER_PATIENT and attempts < max_attempts:
            attempts += 1
            
            # 랜덤한 Z축 슬라이스 및 X, Y 좌표 선택
            z = np.random.randint(0, d)
            x = np.random.randint(0, h - PATCH_SIZE)
            y = np.random.randint(0, w - PATCH_SIZE)

            # 16x16 패치 추출
            patch = data[x:x+PATCH_SIZE, y:y+PATCH_SIZE, z]

            # 배경(빈 공간)이 너무 많은 패치는 제외 (0보다 큰 픽셀이 80% 이상일 때만 사용)
            if np.mean(patch > 0) < 0.8:
                continue
                
            # 패치 정규화 (Intensity 차이에 의한 DC 성분(주파수 0) 폭주 방지)
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            if patch_std == 0:
                continue
            normalized_patch = (patch - patch_mean) / patch_std

            # 3. 2D 푸리에 변환 (FFT) 수행
            # fft2로 변환 후, fftshift를 통해 저주파(중심)부터 고주파(가장자리)로 재배열
            f_transform = np.fft.fft2(normalized_patch)
            f_shift = np.fft.fftshift(f_transform)
            
            # 복소수에서 진폭(Amplitude)만 추출
            amplitude = np.abs(f_shift)

            # 누적
            total_amplitude_spectrum += amplitude
            patient_patches += 1
            valid_patch_count += 1

    # 4. 전체 평균 계산 및 저장
    if valid_patch_count > 0:
        mean_amplitude_spectrum = total_amplitude_spectrum / valid_patch_count
        np.save(OUTPUT_PATH, mean_amplitude_spectrum)
        print(f"\n✅ 성공! 총 {valid_patch_count}개의 패치를 분석했습니다.")
        print(f"✅ 주파수 템플릿이 [{OUTPUT_PATH}]에 저장되었습니다.")
        
        # 간단한 스펙트럼 통계 출력
        print(f"[템플릿 정보] Center(DC/Low Freq): {mean_amplitude_spectrum[PATCH_SIZE//2, PATCH_SIZE//2]:.2f}, "
              f"Edge(High Freq): {mean_amplitude_spectrum[0, 0]:.2f}")
    else:
        print("\n❌ 유효한 패치를 하나도 추출하지 못했습니다. 조건을 확인해주세요.")

if __name__ == "__main__":
    extract_frequency_template()
import os
import urllib.request
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn

# ==========================================
# 1. DnCNN 아키텍처 정의 (KAIR 가중치 맞춤형 구조)
# ==========================================
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        layers = []
        # Layer 1
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Layer 2 ~ 16 (BN 없음, bias=True)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Layer 17
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=True))
        
        # 이름을 'model'로 지정하여 state_dict의 키("model.0.weight" 등)와 일치시킴
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x에서 예측된 노이즈(residual)를 뺌
        out = self.model(x)
        return x - out

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def download_pretrained_dncnn(model_path="dncnn_15.pth"):
    """널리 쓰이는 사전학습된 가우시안 노이즈(sigma=15) DnCNN 가중치 다운로드"""
    if not os.path.exists(model_path):
        print(f"Downloading pre-trained DnCNN weights to {model_path} ...")
        # Kai Zhang(KAIR)의 공식 DnCNN-S (sigma=15) 가중치 URL
        url = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_15.pth"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def min_max_normalize(slice_2d):
    """0~1 사이로 정규화하고 원본 Min, Max를 반환"""
    s_min = slice_2d.min()
    s_max = slice_2d.max()
    if s_max - s_min == 0:
        return slice_2d, s_min, s_max
    norm_slice = (slice_2d - s_min) / (s_max - s_min)
    return norm_slice, s_min, s_max

def denormalize(norm_slice, s_min, s_max):
    """원래의 NIfTI Intensity 범위로 복원"""
    return norm_slice * (s_max - s_min) + s_min

# ==========================================
# 3. 메인 디노이징 파이프라인
# ==========================================
def denoise_nifti(input_nii_path, output_nii_path, device='cuda'):
    print(f"Loading NIfTI file: {input_nii_path}")
    nii = nib.load(input_nii_path)
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine
    header = nii.header

    # 모델 준비
    model = DnCNN(depth=17, image_channels=1).to(device)
    weight_path = download_pretrained_dncnn("dncnn_15.pth")
    
    # 가중치 로드 (구조 불일치 문제 해결됨)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    print("Denoising slices...")
    denoised_data = np.zeros_like(data)
    
    num_slices = data.shape[2]
    
    with torch.no_grad():
        for i in range(num_slices):
            slice_2d = data[:, :, i]
            
            # 배경(0)만 있는 슬라이스는 패스
            if slice_2d.max() == 0:
                denoised_data[:, :, i] = slice_2d
                continue
                
            # --- [추가] 배경 마스크 추출 (0보다 큰 영역) ---
            mask = (slice_2d > 0).astype(np.float32)
            
            # 1. 정규화
            norm_slice, s_min, s_max = min_max_normalize(slice_2d)
            
            # 2. 텐서 변환 (B, C, H, W)
            tensor_slice = torch.from_numpy(norm_slice).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # 3. 모델 추론
            out_tensor = model(tensor_slice)
            
            # 4. 후처리 (0~1 클리핑 및 역정규화)
            out_numpy = out_tensor.squeeze().cpu().numpy()
            out_numpy = np.clip(out_numpy, 0.0, 1.0)
            denoised_slice = denormalize(out_numpy, s_min, s_max)
            
            # --- [추가] 배경 마스크 적용 ---
            denoised_slice = denoised_slice * mask
            
            denoised_data[:, :, i] = denoised_slice
            
            if (i+1) % 20 == 0:
                print(f"  Processed {i+1}/{num_slices} slices")

    # 원본 NIfTI 헤더와 Affine을 그대로 유지하여 저장
    print(f"Saving denoised volume to: {output_nii_path}")
    denoised_nii = nib.Nifti1Image(denoised_data, affine, header)
    nib.save(denoised_nii, output_nii_path)
    print("Done!")

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    input_file = "/data/BraTS20_Degraded_4x/BraTS20_Training_003/BraTS20_Training_003_flair.nii"
    output_file = "/data/BraTS20_Degraded_4x/BraTS20_Training_003/BraTS20_Training_003_flair_denoised.nii.gz"
    
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    denoise_nifti(input_file, output_file, device=device)
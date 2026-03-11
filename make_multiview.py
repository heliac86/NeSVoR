import nibabel as nb
import numpy as np
import os

def create_multiview_stacks(input_nii_path, output_dir="multiview_data"):
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_nii_path).replace(".nii.gz", "")
    
    # 1. 원본 데이터 로드
    img = nb.load(input_nii_path)
    data = img.get_fdata()
    aff = img.affine.copy()  # 4x4 물리 공간 변환 행렬
    
    print(f"Original shape: {data.shape}")
    
    # ---------------------------------------------------------
    # [뷰 1] Axial (원본 그대로. z축이 슬라이스 방향)
    # ---------------------------------------------------------
    path_ax = os.path.join(output_dir, f"{base_name}_axial.nii.gz")
    nb.save(img, path_ax)
    print(f"Saved Axial: {path_ax}")
    
    # ---------------------------------------------------------
    # [뷰 2] Coronal (y축을 슬라이스 방향으로)
    # ---------------------------------------------------------
    # Data 축 변경: (X, Y, Z) -> (X, Z, Y) 
    # 이제 새로운 배열의 3번째 차원(슬라이스 축)은 원래의 Y축 데이터가 됨
    data_cor = np.transpose(data, (0, 2, 1))
    
    # Affine 행렬 열(Column) 스왑: x, y, z -> x, z, y
    aff_cor = aff.copy()
    aff_cor[:, [1, 2]] = aff[:, [2, 1]] 
    
    new_header = img.header.copy()
    new_header.set_data_shape(data_cor.shape)
    # pixdim을 affine column norm에서 계산
    spacings = np.sqrt((aff_cor[:3, :3] ** 2).sum(axis=0))
    new_header['pixdim'][1:4] = spacings
    img_cor = nb.Nifti1Image(data_cor, aff_cor, new_header)
    path_cor = os.path.join(output_dir, f"{base_name}_coronal.nii.gz")
    nb.save(img_cor, path_cor)
    print(f"Saved Coronal: {path_cor} / Shape: {data_cor.shape}")
    
    # ---------------------------------------------------------
    # [뷰 3] Sagittal (x축을 슬라이스 방향으로)
    # ---------------------------------------------------------
    # Data 축 변경: (X, Y, Z) -> (Y, Z, X)
    # 이제 새로운 배열의 3번째 차원(슬라이스 축)은 원래의 X축 데이터가 됨
    data_sag = np.transpose(data, (1, 2, 0))
    
    # Affine 행렬 열(Column) 스왑: x, y, z -> y, z, x
    aff_sag = aff.copy()
    aff_sag[:, [0, 1, 2]] = aff[:, [1, 2, 0]]
    
    img_sag = nb.Nifti1Image(data_sag, aff_sag, img.header)
    path_sag = os.path.join(output_dir, f"{base_name}_sagittal.nii.gz")
    nb.save(img_sag, path_sag)
    print(f"Saved Sagittal: {path_sag} / Shape: {data_sag.shape}")

# 실행 예시 (사용하실 BraTS 훈련 데이터 경로로 수정하세요)
if __name__ == "__main__":
    # 예: "003_flair_01.nii.gz" 를 넣으면 3개의 파일이 생성됩니다.
    # create_multiview_stacks("/data/brats_1027_pre/BraTS20_Training_003/BraTS20_Training_003_flair.nii", "multiview_output")
    create_multiview_stacks("/data/BraTS20_Degraded_4x/BraTS20_Training_003/BraTS20_Training_003_flair.nii", "multiview_4x_output")
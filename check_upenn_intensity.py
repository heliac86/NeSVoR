import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("/data/UPENN-GBM_pre")
OUTPUT_CSV = "UPENN-GBM_pre_intensity_mean.csv"

def compute_folder_intensity(data_dir: Path, output_csv: str):
    all_files = list(data_dir.rglob("*.nii")) + list(data_dir.rglob("*.nii.gz"))
    records = []

    print(f"총 {len(all_files)}개의 파일을 검사합니다.")

    for file_path in tqdm(all_files, desc="측정 중"):
        filename = file_path.name.lower()
        
        # 종양 마스크(seg) 파일은 강도 측정에서 제외
        if "_seg" in filename or "seg." in filename:
            continue

        try:
            img = nib.load(str(file_path))
            data = img.get_fdata(dtype=np.float32)

            # evaluate_batch_v2 기준 뇌 영역 마스크
            brain_mask = data > 0

            if np.any(brain_mask):
                brain_voxels = data[brain_mask]
                mean_val = float(brain_voxels.mean())
                std_val = float(brain_voxels.std())
                voxel_count = int(np.sum(brain_mask))
            else:
                mean_val, std_val, voxel_count = 0.0, 0.0, 0

            records.append({
                "patient_id": file_path.parent.name,
                "filename": file_path.name,
                "voxel_count": voxel_count,
                "intensity_mean": mean_val,
                "intensity_std": std_val
            })

        except Exception as e:
            print(f"\n❌ 에러 ({file_path.name}): {e}")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n결과가 저장되었습니다: {output_csv}")
    print("\n[요약 통계]")
    print(df[['intensity_mean', 'intensity_std']].describe())

if __name__ == "__main__":
    compute_folder_intensity(DATA_DIR, OUTPUT_CSV)
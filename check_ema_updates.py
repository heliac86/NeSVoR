#!/usr/bin/env python3
"""
check_ema_updates.py
slice_sample_counts_main.npy로 슬라이스당 실제 EMA 업데이트 횟수 확인
"""

import os
import numpy as np
import pandas as pd

RECON_ROOT = "/dshome/ddualab/dongnyeok/NeSVoR/recon_analysis"
TEST_CSV   = "/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
MODALITIES = ["flair", "t1ce"]

df_ids   = pd.read_csv(TEST_CSV)
full_ids = df_ids.iloc[:, 0].str.strip().tolist()

rows = []
for full_id in full_ids:
    patient_id  = full_id.split("_")[-1]
    patient_dir = os.path.join(RECON_ROOT, full_id)

    for mod in MODALITIES:
        stem         = f"{patient_id}_{mod}_4x5_analysis"
        analysis_dir = os.path.join(patient_dir, f"{stem}_slice_analysis")
        counts_path  = os.path.join(analysis_dir, "slice_sample_counts_main.npy")
        pixels_path  = os.path.join(analysis_dir, "slice_pixel_counts.npy")

        if not os.path.exists(counts_path):
            continue

        counts = np.load(counts_path).astype(np.float64)   # 슬라이스별 총 샘플링 픽셀 수
        pixels = np.load(pixels_path).astype(np.float64) if os.path.exists(pixels_path) else None

        # 슬라이스당 EMA 업데이트 횟수 = 슬라이스가 배치에 등장한 횟수
        # counts에는 '픽셀 수'가 누적되므로, 슬라이스당 픽셀 수로 나눠야 '배치 등장 횟수'가 됨
        if pixels is not None and pixels.mean() > 0:
            update_counts = counts / (pixels + 1e-8)
        else:
            # 픽셀 수 모를 경우 counts 자체를 사용 (픽셀 단위 참고용)
            update_counts = counts

        rows.append({
            "patient":              patient_id,
            "modality":             mod,
            "n_slices":             len(counts),
            # 슬라이스당 EMA 업데이트 횟수 통계
            "updates_mean":         update_counts.mean(),
            "updates_min":          update_counts.min(),
            "updates_max":          update_counts.max(),
            "updates_p25":          np.percentile(update_counts, 25),
            "updates_p50":          np.percentile(update_counts, 50),
            "updates_p75":          np.percentile(update_counts, 75),
            # 아예 한 번도 업데이트 안 된 슬라이스 수
            "never_updated":        int((update_counts < 0.5).sum()),
            # 원시 픽셀 샘플링 총량 (참고용)
            "total_pixel_samples":  counts.sum(),
        })

df = pd.DataFrame(rows)

print("=" * 65)
print("슬라이스당 EMA 업데이트 횟수 요약")
print("=" * 65)
for mod in MODALITIES:
    sub = df[df["modality"] == mod]
    print(f"\n[{mod.upper()}]  n_cases={len(sub)}")
    print(f"  updates/slice  mean={sub['updates_mean'].mean():.1f}  "
          f"min={sub['updates_min'].min():.1f}  "
          f"max={sub['updates_max'].max():.1f}")
    print(f"  median(p50)    {sub['updates_p50'].mean():.1f}")
    print(f"  never_updated  avg {sub['never_updated'].mean():.1f} slices/case")

print("\n[alpha별 초기값 잔존율 — 위 mean 업데이트 횟수 기준]")
mean_k = df["updates_mean"].mean()
print(f"  k (평균 업데이트 횟수) = {mean_k:.1f}")
for alpha in [0.80, 0.90, 0.95, 0.99]:
    retention = alpha ** mean_k
    print(f"  alpha={alpha}  →  잔존율 = {retention:.3f}  ({retention*100:.1f}%)")

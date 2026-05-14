#!/usr/bin/env python3
# experiment_B_fft.py
# 실험 B: 3D FFT 기반 주파수 대역별 복원 오차 분해
# 비교: Ours_O vs no_gating vs Trilinear

import os, re, glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import ndimage, stats
from typing import Optional

# ================================================================
# 경로 설정
# ================================================================
GT_ROOT        = "/data/BraTS2020_TrainingData"
OURS_ROOT      = "/dshome/ddualab/dongnyeok/NeSVoR/recon_gating_hsm_O/hsm_O"
NOGATING_ROOT  = "/dshome/ddualab/dongnyeok/NeSVoR/recon_gating_hsm_O/no_gating"
TRILINEAR_ROOT = "/dshome/ddualab/dongnyeok/densenet121/trilinear"
TEST_CSV       = "/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
OUT_DIR        = "/dshome/ddualab/dongnyeok/NeSVoR/experiment_B"
os.makedirs(OUT_DIR, exist_ok=True)

MODALITIES = ["flair", "t1ce", "t2", "t1"]

# ================================================================
# 주파수 밴드 정의 (인수인계서 레벨별 Nyquist 기준, cyc/voxel @ 1mm iso)
# ================================================================
# 각 레벨의 Nyquist (cyc/mm) = 격자 간격의 역수 / 2
# L1:0.031, L2:0.042, L3:0.058, L4:0.081, L5:0.113,
# L6:0.156, L7:0.217, L8:0.300, L9:0.415, L10:0.573, L11:0.792, L12:1.096→clip 0.5
BANDS = [
    {"name": "B1\n(L1-L3)",  "label": "B1: L1-L3\n(초저주파\n~32mm)",   "f_lo": 0.000, "f_hi": 0.058},
    {"name": "B2\n(L4-L5)",  "label": "B2: L4-L5\n(저중간\n8.9-17mm)",  "f_lo": 0.058, "f_hi": 0.113},
    {"name": "B3\n(L6)",     "label": "B3: L6\n(6.4-8.9mm)",            "f_lo": 0.113, "f_hi": 0.156},
    {"name": "B4\n(L7)★",   "label": "B4: L7★\n(4.6-6.4mm)\n피크",     "f_lo": 0.156, "f_hi": 0.217},
    {"name": "B5\n(L8)",     "label": "B5: L8\n(3.3-4.6mm)",            "f_lo": 0.217, "f_hi": 0.300},
    {"name": "B6\n(L9-L10)", "label": "B6: L9-L10\n(1.7-3.3mm)",        "f_lo": 0.300, "f_hi": 0.573},
    {"name": "B7\n(L11-L12)","label": "B7: L11-L12\n(초고주파\n<1.3mm)","f_lo": 0.573, "f_hi": 0.500},
]
# B7은 Nyquist(0.5) 초과 정의이므로 실제로는 0.5까지만 에너지 존재 → f_hi=0.500으로 cap
BANDS[-1]["f_hi"] = 0.500
N_BANDS = len(BANDS)

# ================================================================
# 유틸 함수 (제공된 함수 그대로 사용)
# ================================================================
def find_modality_file(dataset_root: str, patient_id: str, modality: str) -> Optional[str]:
    patient_dir = os.path.join(dataset_root, patient_id)
    if not os.path.isdir(patient_dir):
        return None
    m = re.search(r'(\d+)$', patient_id)
    short_id = m.group(1) if m else ""
    patterns = [
        os.path.join(patient_dir, f"*{patient_id}*{modality}*.nii*"),
        os.path.join(patient_dir, f"*{short_id}*{modality}*.nii*"),
        os.path.join(patient_dir, f"*{modality}*{short_id}*.nii*"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def robust_align_recon(orig_nib, recon_nib):
    orig_shape = orig_nib.shape[:3]
    orig_ornt  = nib.io_orientation(orig_nib.affine)
    recon_ornt = nib.io_orientation(recon_nib.affine)
    if not np.array_equal(orig_ornt, recon_ornt):
        transform = nib.ornt_transform(recon_ornt, orig_ornt)
        recon_nib = recon_nib.as_reoriented(transform)
    recon_shape = recon_nib.shape[:3]
    recon_data  = recon_nib.get_fdata().astype(np.float32)
    if orig_shape != recon_shape:
        inv_orig_affine       = np.linalg.inv(orig_nib.affine)
        recon_origin_physical = recon_nib.affine @ np.array([0, 0, 0, 1])
        offset_voxels         = inv_orig_affine @ recon_origin_physical
        x_off, y_off, z_off   = np.round(offset_voxels[:3]).astype(int)
        rx, ry, rz = recon_shape
        aligned_data = np.zeros(orig_shape, dtype=np.float32)
        tgt_x0, tgt_x1 = max(0, x_off),  min(orig_shape[0], x_off + rx)
        tgt_y0, tgt_y1 = max(0, y_off),  min(orig_shape[1], y_off + ry)
        tgt_z0, tgt_z1 = max(0, z_off),  min(orig_shape[2], z_off + rz)
        src_x0 = max(0, -x_off); src_x1 = src_x0 + (tgt_x1 - tgt_x0)
        src_y0 = max(0, -y_off); src_y1 = src_y0 + (tgt_y1 - tgt_y0)
        src_z0 = max(0, -z_off); src_z1 = src_z0 + (tgt_z1 - tgt_z0)
        if (tgt_x1 > tgt_x0) and (tgt_y1 > tgt_y0) and (tgt_z1 > tgt_z0):
            aligned_data[tgt_x0:tgt_x1, tgt_y0:tgt_y1, tgt_z0:tgt_z1] = \
                recon_data[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
        recon_data = aligned_data
    new_header = orig_nib.header.copy()
    new_header.set_data_shape(recon_data.shape)
    new_header['scl_slope'] = 1.0
    new_header['scl_inter'] = 0.0
    return nib.Nifti1Image(recon_data, orig_nib.affine, new_header)


def normalize_pair(orig_array, recon_array, brain_mask):
    k = (np.sum(orig_array[brain_mask] * recon_array[brain_mask]) /
         (np.sum(recon_array[brain_mask] ** 2) + 1e-8))
    recon_matched = recon_array * k
    mn = np.percentile(orig_array[brain_mask], 0.1)
    mx = np.percentile(orig_array[brain_mask], 99.9)
    orig_norm  = np.clip((orig_array    - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    recon_norm = np.clip((recon_matched - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    orig_norm[~brain_mask]  = 0.0
    recon_norm[~brain_mask] = 0.0
    return orig_norm, recon_norm, float(k)

# ================================================================
# 3D 주파수 밴드 필터링 (Gaussian bandpass)
# ================================================================
def make_freq_grid(shape):
    """각 축의 주파수 좌표 생성 (0~0.5 cyc/voxel, 1mm iso → cyc/mm)"""
    grids = [np.fft.fftfreq(s) for s in shape]   # 범위: -0.5 ~ 0.5
    fx, fy, fz = np.meshgrid(*grids, indexing='ij')
    return np.sqrt(fx**2 + fy**2 + fz**2)         # 3D 방사형 주파수


def bandpass_filter_3d(volume, f_lo, f_hi, freq_grid, sigma_ratio=0.3):
    """
    Gaussian-windowed bandpass 필터.
    sigma_ratio: 각 경계에서의 transition 폭 (밴드 폭 대비)
    f_hi=0.5 이상이면 lowpass (고주파 밴드용)
    """
    bw = f_hi - f_lo
    sigma = max(bw * sigma_ratio, 0.005)   # 최소 sigma 보장

    F = np.fft.fftn(volume)

    if f_lo <= 0.0:
        # 저역통과 성분: f_hi 미만
        H = 1.0 - 0.5 * (1.0 + np.tanh((freq_grid - f_hi) / sigma))
    elif f_hi >= 0.499:
        # 고역통과 성분: f_lo 이상
        H = 0.5 * (1.0 + np.tanh((freq_grid - f_lo) / sigma))
    else:
        # 대역통과
        H_hi = 1.0 - 0.5 * (1.0 + np.tanh((freq_grid - f_hi) / sigma))
        H_lo = 0.5 * (1.0 + np.tanh((freq_grid - f_lo) / sigma))
        H = H_lo * H_hi

    filtered = np.real(np.fft.ifftn(F * H))
    return filtered.astype(np.float32)


def band_rmse(orig, recon, freq_grid, f_lo, f_hi):
    """특정 주파수 밴드에서의 RMSE (brain mask 불필요 — 주파수 도메인 비교)"""
    orig_band  = bandpass_filter_3d(orig,  f_lo, f_hi, freq_grid)
    recon_band = bandpass_filter_3d(recon, f_lo, f_hi, freq_grid)
    diff = orig_band - recon_band
    return float(np.sqrt(np.mean(diff**2)))

# ================================================================
# 메인 루프
# ================================================================
records = []

full_ids = pd.read_csv(TEST_CSV)["patient_id"].tolist()   # 컬럼명 확인 필요
print(f"총 케이스: {len(full_ids)} × {len(MODALITIES)} = {len(full_ids)*len(MODALITIES)}")

for full_id in full_ids:
    m = re.search(r'(\d+)$', full_id)
    short_id = m.group(1) if m else full_id

    gt_nib = None
    for mod in MODALITIES:
        # ── GT 로드 ──────────────────────────────────────────────
        gt_path = find_modality_file(GT_ROOT, full_id, mod)
        if gt_path is None:
            print(f"  [SKIP] GT 없음: {full_id} / {mod}")
            continue
        gt_nib = nib.load(gt_path)
        gt_arr = gt_nib.get_fdata().astype(np.float32)
        brain_mask = gt_arr > 0

        # 주파수 그리드는 GT shape 기준 (케이스/모달 공통 shape이면 캐싱 가능)
        freq_grid = make_freq_grid(gt_arr.shape)

        # ── 각 조건별 볼륨 로드 및 전처리 ───────────────────────
        conditions = {}

        # Ours_O
        ours_fname = os.path.join(OURS_ROOT, full_id, f"{short_id}_{mod}_gating.nii.gz")
        if os.path.exists(ours_fname):
            ours_nib = robust_align_recon(gt_nib, nib.load(ours_fname))
            _, ours_norm, _ = normalize_pair(gt_arr, ours_nib.get_fdata().astype(np.float32), brain_mask)
            conditions["ours"] = ours_norm
        else:
            print(f"  [MISS] Ours: {ours_fname}")

        # no_gating
        nog_fname = os.path.join(NOGATING_ROOT, full_id, f"{short_id}_{mod}_gating.nii.gz")
        if os.path.exists(nog_fname):
            nog_nib = robust_align_recon(gt_nib, nib.load(nog_fname))
            _, nog_norm, _ = normalize_pair(gt_arr, nog_nib.get_fdata().astype(np.float32), brain_mask)
            conditions["no_gating"] = nog_norm
        else:
            print(f"  [MISS] no_gating: {nog_fname}")

        # Trilinear
        tri_path = find_modality_file(TRILINEAR_ROOT, full_id, mod)
        if tri_path:
            tri_nib = robust_align_recon(gt_nib, nib.load(tri_path))
            _, tri_norm, _ = normalize_pair(gt_arr, tri_nib.get_fdata().astype(np.float32), brain_mask)
            conditions["trilinear"] = tri_norm
        else:
            print(f"  [MISS] Trilinear: {full_id}/{mod}")

        # GT도 normalize (normalize_pair는 recon 기준 — GT 자체 정규화)
        mn = np.percentile(gt_arr[brain_mask], 0.1)
        mx = np.percentile(gt_arr[brain_mask], 99.9)
        gt_norm = np.clip((gt_arr - mn) / (mx - mn + 1e-8), 0.0, 1.0)
        gt_norm[~brain_mask] = 0.0

        # ── 밴드별 RMSE 계산 ─────────────────────────────────────
        rec = {"patient": short_id, "full_id": full_id, "modality": mod}
        for i, band in enumerate(BANDS):
            f_lo, f_hi = band["f_lo"], band["f_hi"]
            for cond_name, cond_arr in conditions.items():
                rmse = band_rmse(gt_norm, cond_arr, freq_grid, f_lo, f_hi)
                rec[f"B{i+1}_{cond_name}"] = rmse

        records.append(rec)
        print(f"  ✓ {full_id} / {mod} 완료")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "expB_band_rmse_raw.csv"), index=False)
print(f"\n원시 데이터 저장 완료: {len(df)} rows")

# ================================================================
# 집계: 모달리티 평균 → 밴드별 RMSE 평균±std
# ================================================================
band_names = [f"B{i+1}" for i in range(N_BANDS)]
cond_names = ["ours", "no_gating", "trilinear"]
COLORS = {"ours": "#4C72B0", "no_gating": "#DD8452", "trilinear": "#55A868"}
LABELS = {"ours": "Ours_O", "no_gating": "no_gating", "trilinear": "Trilinear"}

agg = {}
for cond in cond_names:
    agg[cond] = {"mean": [], "std": []}
    for i in range(N_BANDS):
        col = f"B{i+1}_{cond}"
        if col in df.columns:
            agg[cond]["mean"].append(df[col].mean())
            agg[cond]["std"].append(df[col].std())
        else:
            agg[cond]["mean"].append(np.nan)
            agg[cond]["std"].append(np.nan)

# ================================================================
# 시각화 1: 밴드별 RMSE 비교 바차트
# ================================================================
x = np.arange(N_BANDS)
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
for j, cond in enumerate(cond_names):
    means = np.array(agg[cond]["mean"])
    stds  = np.array(agg[cond]["std"])
    ax.bar(x + j*width, means, width, yerr=stds, capsize=3,
           color=COLORS[cond], alpha=0.85, label=LABELS[cond],
           error_kw={"elinewidth": 1.2})

ax.set_xticks(x + width)
ax.set_xticklabels([b["name"] for b in BANDS], fontsize=10)
ax.set_xlabel("Frequency Band (Hash Grid Level Correspondence)")
ax.set_ylabel("Band RMSE (normalized intensity)")
ax.set_title("Experiment B: Band-wise Reconstruction Error (Ours_O vs no_gating vs Trilinear)")
ax.legend()

# L7 밴드(B4) 강조
ax.axvspan(x[3]-0.2, x[3]+0.95, color="yellow", alpha=0.15, zorder=0, label="_B4 highlight")
ax.text(x[3]+0.35, ax.get_ylim()[1]*0.95, "L7 피크 대역", ha="center", fontsize=9, color="goldenrod")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figB1_band_rmse_bar.png"), dpi=150)
plt.close()

# ================================================================
# 시각화 2: Ours_O 대비 오차 감소율 (no_gating - ours) / no_gating
# ================================================================
fig, ax = plt.subplots(figsize=(12, 5))
reduction_nog = []
reduction_tri = []
for i in range(N_BANDS):
    r_ours = agg["ours"]["mean"][i]
    r_nog  = agg["no_gating"]["mean"][i]
    r_tri  = agg["trilinear"]["mean"][i]
    reduction_nog.append((r_nog - r_ours) / (r_nog + 1e-8) * 100)
    reduction_tri.append((r_tri - r_ours) / (r_tri + 1e-8) * 100)

ax.bar(x - 0.2, reduction_nog, 0.35, color=COLORS["no_gating"], alpha=0.85, label="no_gating → Ours_O 감소율")
ax.bar(x + 0.2, reduction_tri, 0.35, color=COLORS["trilinear"], alpha=0.85, label="Trilinear → Ours_O 감소율")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvspan(x[3]-0.4, x[3]+0.5, color="yellow", alpha=0.15, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([b["name"] for b in BANDS], fontsize=10)
ax.set_xlabel("Frequency Band")
ax.set_ylabel("Error Reduction Rate vs Ours_O (%)\n(양수 = Ours_O가 더 우수)")
ax.set_title("Band-wise Error Reduction: Ours_O vs no_gating / Trilinear")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figB2_error_reduction.png"), dpi=150)
plt.close()

# ================================================================
# 시각화 3: 모달리티별 B4(L7) RMSE 비교 (Paired)
# ================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
for ax, mod in zip(axes, MODALITIES):
    sub = df[df["modality"] == mod]
    for cond, color in zip(cond_names, [COLORS[c] for c in cond_names]):
        col = f"B4_{cond}"
        if col in sub.columns:
            vals = sub[col].values
            ax.boxplot([vals], positions=[cond_names.index(cond)],
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.6),
                       medianprops=dict(color="red", linewidth=1.5),
                       widths=0.5)
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels([LABELS[c] for c in cond_names], fontsize=8, rotation=15)
    ax.set_title(f"{mod}\nB4(L7) RMSE")
    ax.set_ylabel("Band RMSE" if mod == "flair" else "")
fig.suptitle("B4 (L7 Band) RMSE by Modality", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figB3_b4_by_modality.png"), dpi=150)
plt.close()

# ================================================================
# Step: Paired Wilcoxon 검정 (Ours_O vs no_gating, 밴드별)
# ================================================================
wilcoxon_rows = []
for i in range(N_BANDS):
    col_o = f"B{i+1}_ours"
    col_n = f"B{i+1}_no_gating"
    col_t = f"B{i+1}_trilinear"
    if col_o in df.columns and col_n in df.columns:
        sub = df[[col_o, col_n, col_t]].dropna()
        stat_on, p_on = stats.wilcoxon(sub[col_o], sub[col_n])
        stat_ot, p_ot = stats.wilcoxon(sub[col_o], sub[col_t])
        wilcoxon_rows.append({
            "band": f"B{i+1}",
            "level_range": BANDS[i]["name"].replace("\n", " "),
            "W_ours_vs_nogating": round(stat_on, 2),
            "p_ours_vs_nogating": round(p_on, 6),
            "sig_ours_vs_nogating": "YES" if p_on < 0.05 else "NO",
            "W_ours_vs_trilinear": round(stat_ot, 2),
            "p_ours_vs_trilinear": round(p_ot, 6),
            "sig_ours_vs_trilinear": "YES" if p_ot < 0.05 else "NO",
        })

wil_df = pd.DataFrame(wilcoxon_rows)
wil_df.to_csv(os.path.join(OUT_DIR, "expB_wilcoxon.csv"), index=False)
print("\nWilcoxon 결과:")
print(wil_df.to_string(index=False))

print(f"\n실험 B 완료. 결과 폴더: {OUT_DIR}")

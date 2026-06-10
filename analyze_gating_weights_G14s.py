# analyze_gating_weights_G14s.py
# 목적: G14s GatingMLP 가중치 분석
#   - 전역 가중치: 볼륨 좌표 전체 순회 → mean(dim=0) → 레벨별 평균
#   - 공간 분포: 레벨별 axial 슬라이스 히트맵 (선택적 저장)
#
# 출력 파일:
#   gating_weights_raw.csv          ← 케이스별 레벨 평균 가중치 (전역)
#   gating_stats_summary.csv        ← 레벨별 통계 요약
#   gating_kruskal.csv              ← 모달리티 간 Kruskal-Wallis 검정
#   fig1_overall_bar.png            ← 전체 평균±std 바차트
#   fig2_overall_boxplot.png        ← 전체 박스플롯
#   fig3_modality_profile.png       ← 모달리티별 평균 프로파일
#   fig4_heatmap.png                ← 케이스×레벨 히트맵
#   fig5_cv.png                     ← 레벨별 CV
#   spatial/                        ← (SAVE_SPATIAL=True 시)
#     {patient}_{modality}_L{k}_axial.png  ← 레벨 k 공간 가중치 axial 슬라이스

import sys, os, re, glob, argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 경로 ──────────────────────────────────────────────────────
RECON_ROOT = "/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s"
OUT_DIR    = "/dshome/ddualab/dongnyeok/NeSVoR/gating_analysis_G14s_2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 실험 파라미터 ──────────────────────────────────────────────
N_LEVELS   = 12
MODALITIES = ["flair", "t1ce", "t1", "t2"]
EXP_TAG    = "G14s"

# 공간 분포 히트맵 저장 여부 (케이스당 N_LEVELS장 → 용량 큼)
SAVE_SPATIAL     = True
# SPATIAL_LEVELS   = [4, 6, 9, 12]   # 저장할 레벨 인덱스 (1-based)
SPATIAL_LEVELS = list(range(1, N_LEVELS + 1))  # N_LEVELS 변경 시 자동 대응
SPATIAL_N_CASES  = 5                 # 모달리티당 저장할 케이스 수 (0=전체)

# 볼륨 좌표 샘플링 stride (클수록 빠르지만 해상도 낮음)
GRID_STRIDE = 4

COLORS = {
    "flair": "#4C72B0",
    "t1ce" : "#DD8452",
    "t2"   : "#55A868",
    "t1"   : "#C44E52",
}

# ================================================================
# 모델 로드 유틸
# ================================================================
def load_inr_from_ckpt(pt_path, device):
    from nesvor.inr.models import INR  # INR 클래스만 임포트
  
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    
    args = ckpt["args"]
    sd   = ckpt["model"]
    bb   = sd["bounding_box"].to(device)  # state_dict 안에 있음
    
    model = INR(bb, args).to(device)          # INR만 생성
    
    # ── [추가된 크기 불일치 해결 로직] ──────────────────────────────
    model_sd = model.state_dict()
    keys_to_delete = []
    
    for key in sd.keys():
        if key in model_sd and sd[key].shape != model_sd[key].shape:
            # 크기가 다른 파라미터(예: encoding.params)를 감지하여 로그를 남기고 제거 명단에 추가
            print(f"    [INFO] 크기 불일치 감지 파라미터 제외 -> {key}: model{model_sd[key].shape} vs ckpt{sd[key].shape}")
            keys_to_delete.append(key)
            
    for key in keys_to_delete:
        del sd[key]  # 체크포인트 데이터에서 해당 파라미터 삭제 (Missing Key 상태로 만듦)
    # ─────────────────────────────────────────────────────────────
    
    # 이제 strict=False가 누락된 encoding.params를 에러 없이 무시하고 넘어가 줍니다.
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def extract_global_weights(model, nii_path: str, device,
                            stride: int = 4):
    """
    볼륨 전체 좌표를 stride 간격으로 샘플링하여
    GatingMLP에 통과 → 평균 가중치 벡터 (n_levels,) 반환.
    공간 분포 볼륨도 함께 반환 (H//stride, W//stride, D//stride, n_levels).
    """
    nib_img  = nib.load(nii_path)
    shape    = nib_img.header.get_data_shape()[:3]
    affine   = nib_img.affine

    H, W, D = shape
    xs = np.arange(0, H, stride)
    ys = np.arange(0, W, stride)
    zs = np.arange(0, D, stride)

    # voxel → world 좌표
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)  # (h,w,d,3)
    grid_flat = grid.reshape(-1, 3).astype(np.float32)

    # affine 적용: ijk → RAS mm
    ones  = np.ones((grid_flat.shape[0], 1), dtype=np.float32)
    grid_h = np.concatenate([grid_flat, ones], axis=1)           # (N, 4)
    xyz_mm = (affine @ grid_h.T).T[:, :3].astype(np.float32)     # (N, 3)

    xyz_t = torch.from_numpy(xyz_mm).to(device)

    # 배치 단위 추론 (메모리 절약)
    BATCH = 65536
    all_weights = []
    with torch.no_grad():
        for i in range(0, xyz_t.shape[0], BATCH):
            batch_xyz = xyz_t[i:i+BATCH]
            _, w = model(batch_xyz)          # (B, n_levels)
            if w is not None:
                all_weights.append(w.cpu().float())

    if not all_weights:
        return None, None

    weights_flat = torch.cat(all_weights, dim=0)  # (N, n_levels)
    global_mean  = weights_flat.mean(dim=0).numpy()  # (n_levels,)

    # 공간 분포 볼륨 reshape
    h_s = len(xs); w_s = len(ys); d_s = len(zs)
    weights_vol = weights_flat.numpy().reshape(h_s, w_s, d_s, -1)  # (h,w,d,n_lev)

    return global_mean, weights_vol


# ================================================================
# Step 1: 전체 pt 파일에서 가중치 추출 → DataFrame
# ================================================================

# 파일명 패턴: {short_id}_{modality}_4x5_G14s.pt
# 예: 003_flair_4x5_G14s.pt
pt_files = sorted(glob.glob(f"{RECON_ROOT}/**/*_{EXP_TAG}.pt", recursive=True))
print(f"총 pt 파일 수: {len(pt_files)}")

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}")

records        = []
spatial_cases  = {mod: 0 for mod in MODALITIES}
spatial_dir    = os.path.join(OUT_DIR, "spatial")
if SAVE_SPATIAL:
    os.makedirs(spatial_dir, exist_ok=True)

for pt_path in pt_files:
    fname = os.path.basename(pt_path)
    # 예: 003_flair_4x5_G14s.pt
    m = re.match(r"(\d+)_(\w+)_4x5_G14s\.pt", fname)
    if not m:
        print(f"  [SKIP] 파일명 파싱 실패: {fname}")
        continue
    patient_id, modality = m.group(1), m.group(2)
    if modality not in MODALITIES:
        print(f"  [SKIP] 알 수 없는 모달리티: {modality}")
        continue

    # 대응 nii.gz 경로 추정
    nii_path = pt_path.replace(".pt", ".nii.gz")
    if not os.path.exists(nii_path):
        print(f"  [WARN] nii.gz 없음, 좌표는 bounding_box 기반으로 대체: {fname}")
        nii_path = None

    print(f"  처리 중: {patient_id} / {modality}")
    try:
        model = load_inr_from_ckpt(pt_path, device)

        if nii_path:
            global_mean, weights_vol = extract_global_weights(
                model, nii_path, device, stride=GRID_STRIDE
            )
        else:
            # nii 없을 경우: bounding_box 격자 샘플링 fallback
            bb     = model.bounding_box.cpu().numpy()
            xs_mm  = np.linspace(bb[0,0], bb[1,0], 30)
            ys_mm  = np.linspace(bb[0,1], bb[1,1], 30)
            zs_mm  = np.linspace(bb[0,2], bb[1,2], 30)
            grid   = np.stack(np.meshgrid(xs_mm, ys_mm, zs_mm, indexing='ij'), -1)
            xyz_t  = torch.from_numpy(grid.reshape(-1,3).astype(np.float32)).to(device)
            with torch.no_grad():
                _, w = model(xyz_t)
            global_mean  = w.cpu().float().mean(dim=0).numpy() if w is not None else np.ones(N_LEVELS)
            weights_vol  = None

    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        continue

    if global_mean is None:
        print(f"  [WARN] gating_weights None (use_gating=False?): {fname}")
        global_mean = np.ones(N_LEVELS)

    rec = {"patient": patient_id, "modality": modality}
    for lvl in range(N_LEVELS):
        rec[f"L{lvl+1:02d}"] = float(global_mean[lvl])
    records.append(rec)

    # ── 공간 분포 저장 ──────────────────────────────────────────
    if SAVE_SPATIAL and weights_vol is not None:
        if SPATIAL_N_CASES == 0 or spatial_cases[modality] < SPATIAL_N_CASES:
            mid_z = weights_vol.shape[2] // 2
            for lv in SPATIAL_LEVELS:
                if lv > weights_vol.shape[3]:
                    continue
                w_slice = weights_vol[:, :, mid_z, lv - 1]  # (h, w)
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(w_slice.T, cmap="hot", origin="lower",
                               vmin=0.0, vmax=weights_vol[:,:,:,lv-1].max())
                plt.colorbar(im, ax=ax, label="Gating Weight")
                ax.set_title(f"{patient_id}_{modality} | Level L{lv:02d} | Axial mid")
                ax.axis("off")
                plt.tight_layout()
                save_name = f"{patient_id}_{modality}_L{lv:02d}_axial.png"
                plt.savefig(os.path.join(spatial_dir, save_name), dpi=120)
                plt.close()
            spatial_cases[modality] += 1

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

# ================================================================
# DataFrame 구성
# ================================================================
df = pd.DataFrame(records)
level_cols   = [f"L{i:02d}" for i in range(1, N_LEVELS + 1)]
level_labels = [f"L{i}"    for i in range(1, N_LEVELS + 1)]

print(f"\n추출 완료: {len(df)} 케이스")
print(df["modality"].value_counts().to_string())
df.to_csv(os.path.join(OUT_DIR, "gating_weights_raw.csv"), index=False)
print("gating_weights_raw.csv 저장 완료")

# ================================================================
# Step 2: 레벨별 통계 요약
# ================================================================
stats_rows = []
for lvl in level_cols:
    for mod in MODALITIES + ["ALL"]:
        sub = df[df["modality"] == mod][lvl] if mod != "ALL" else df[lvl]
        stats_rows.append({
            "level"    : lvl,
            "modality" : mod,
            "mean"     : sub.mean(),
            "std"      : sub.std(),
            "median"   : sub.median(),
            "cv"       : sub.std() / sub.mean() if sub.mean() != 0 else np.nan,
            "n"        : len(sub),
        })
pd.DataFrame(stats_rows).to_csv(
    os.path.join(OUT_DIR, "gating_stats_summary.csv"), index=False)
print("gating_stats_summary.csv 저장 완료")

# ================================================================
# Step 3: Kruskal-Wallis 검정
# ================================================================
kw_rows = []
for lvl in level_cols:
    groups = [df[df["modality"] == m][lvl].values for m in MODALITIES
              if len(df[df["modality"] == m]) > 1]
    if len(groups) < 2:
        continue
    h_stat, p_val = stats.kruskal(*groups)
    kw_rows.append({
        "level"      : lvl,
        "H_stat"     : round(h_stat, 4),
        "p_value"    : round(p_val, 6),
        "significant": "YES" if p_val < 0.05 else "NO",
    })
kw_df = pd.DataFrame(kw_rows)
kw_df.to_csv(os.path.join(OUT_DIR, "gating_kruskal.csv"), index=False)
print("gating_kruskal.csv 저장 완료")
print("\nKruskal-Wallis 결과:")
print(kw_df.to_string(index=False))

# ================================================================
# 시각화 공통 설정
# ================================================================
plt.rcParams.update({
    "font.family"   : "DejaVu Sans",
    "font.size"     : 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi"    : 150,
})
x = np.arange(N_LEVELS)

# ================================================================
# Fig 1: 전체 평균±std 바차트
# ================================================================
all_mean = df[level_cols].mean().values
all_std  = df[level_cols].std().values

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x, all_mean, yerr=all_std, capsize=4,
       color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.8,
       error_kw={"ecolor": "#333333", "elinewidth": 1.2})
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2,
           alpha=0.7, label="Uniform (1.0)")
ax.set_xticks(x); ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Gating Weight (mean over volume)")
ax.set_title(f"Overall Level Gating Weights — G14s All Cases (mean ± std)")
ax.legend(); ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_overall_bar.png"), dpi=150)
plt.close(); print("fig1_overall_bar.png 저장 완료")

# ================================================================
# Fig 2: 전체 박스플롯
# ================================================================
fig, ax = plt.subplots(figsize=(12, 5))
bp = ax.boxplot(
    [df[lvl].values for lvl in level_cols],
    labels=level_labels, patch_artist=True,
    medianprops={"color": "red", "linewidth": 1.5},
    whiskerprops={"linewidth": 1.2},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
)
for patch in bp["boxes"]:
    patch.set(facecolor="#4C72B0", alpha=0.6)
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2,
           alpha=0.7, label="Uniform (1.0)")
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Gating Weight")
ax.set_title(f"Level Gating Weight Distribution — G14s All Cases")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_overall_boxplot.png"), dpi=150)
plt.close(); print("fig2_overall_boxplot.png 저장 완료")

# ================================================================
# Fig 3: 모달리티별 평균 프로파일
# ================================================================
fig, ax = plt.subplots(figsize=(11, 5))
for mod in MODALITIES:
    sub  = df[df["modality"] == mod]
    mean = sub[level_cols].mean().values
    std  = sub[level_cols].std().values
    ax.plot(x, mean, marker="o", markersize=5, linewidth=2,
            color=COLORS[mod], label=f"{mod} (n={len(sub)})")
    ax.fill_between(x, mean - std, mean + std,
                    color=COLORS[mod], alpha=0.12)
ax.axhline(y=1.0, color="gray", linestyle="--",
           linewidth=1.0, alpha=0.6, label="Uniform (1.0)")
ax.set_xticks(x); ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Gating Weight (mean over volume)")
ax.set_title("Gating Weight Profile by Modality (G14s) — mean ± std shading")
ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_modality_profile.png"), dpi=150)
plt.close(); print("fig3_modality_profile.png 저장 완료")

# ================================================================
# Fig 4: 케이스×레벨 히트맵
# ================================================================
df_sorted    = df.sort_values(["modality", "patient"])
heatmap_data = df_sorted[level_cols].values

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn",
               vmin=0.5, vmax=1.6)
plt.colorbar(im, ax=ax, label="Gating Weight (mean over volume)")

mod_counts = df_sorted["modality"].value_counts()[MODALITIES].values
y_offset   = 0
for mod, cnt in zip(MODALITIES, mod_counts):
    ax.axhline(y=y_offset - 0.5, color="white", linewidth=2)
    ax.text(-0.8, y_offset + cnt / 2, mod, ha="right", va="center",
            fontsize=11, fontweight="bold", color=COLORS[mod])
    y_offset += cnt

ax.set_xticks(np.arange(N_LEVELS)); ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Cases (grouped by modality)")
ax.set_title(f"Gating Weights Heatmap — G14s All Cases × {N_LEVELS} Levels")
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_heatmap.png"), dpi=150)
plt.close(); print("fig4_heatmap.png 저장 완료")

# ================================================================
# Fig 5: 레벨별 CV
# ================================================================
fig, ax = plt.subplots(figsize=(11, 4))
for mod in MODALITIES:
    sub = df[df["modality"] == mod]
    cv  = (sub[level_cols].std() / sub[level_cols].mean()).values
    ax.plot(x, cv, marker="s", markersize=4, linewidth=1.5,
            color=COLORS[mod], label=mod)
all_cv = (df[level_cols].std() / df[level_cols].mean()).values
ax.plot(x, all_cv, marker="D", markersize=5, linewidth=2,
        color="black", linestyle="--", label="ALL")
ax.set_xticks(x); ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level")
ax.set_ylabel("CV (std / mean)")
ax.set_title("Coefficient of Variation per Level (G14s) — Lower = More Consistent")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_cv.png"), dpi=150)
plt.close(); print("fig5_cv.png 저장 완료")

print(f"\n모든 분석 완료. 결과 폴더: {OUT_DIR}")

# analyze_gating_weights.py
# 목적: 192케이스 level_weights 추출 → 전체/모달리티별 분석 + 시각화
#
# 출력 파일:
#   gating_weights_raw.csv        ← 케이스별 raw 가중치
#   fig1_overall_bar.png          ← 전체 평균±std 바차트
#   fig2_overall_boxplot.png      ← 전체 박스플롯
#   fig3_modality_profile.png     ← 모달리티별 평균 프로파일
#   fig4_heatmap.png              ← 케이스×레벨 히트맵
#   fig5_cv.png                   ← 레벨별 CV (일관성)
#   gating_stats_summary.csv      ← 레벨별 통계 요약
#   gating_kruskal.csv            ← 모달리티 간 Kruskal-Wallis 검정 결과

import torch
import torch.nn.functional as F
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 경로 ──────────────────────────────────────────────────────
RECON_ROOT = "/dshome/ddualab/dongnyeok/NeSVoR/recon_gating"
OUT_DIR    = "/dshome/ddualab/dongnyeok/NeSVoR/gating_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

N_LEVELS   = 12
MODALITIES = ["flair", "t1ce", "t2", "t1"]

# ── 색상 팔레트 (모달리티 4개) ─────────────────────────────────
COLORS = {
    "flair": "#4C72B0",
    "t1ce" : "#DD8452",
    "t2"   : "#55A868",
    "t1"   : "#C44E52",
}

# ================================================================
# Step 1: 전체 pt 파일에서 가중치 추출 → DataFrame
# ================================================================
pt_files = sorted(glob.glob(f"{RECON_ROOT}/**/*_model.pt", recursive=True))
print(f"총 pt 파일 수: {len(pt_files)}")

records = []
for pt_path in pt_files:
    fname = os.path.basename(pt_path)
    # 파일명: 003_flair_gating_model.pt
    m = re.match(r"(\d+)_(\w+)_gating_model\.pt", fname)
    if not m:
        print(f"  [SKIP] 파일명 파싱 실패: {fname}")
        continue
    patient_id, modality = m.group(1), m.group(2)

    try:
        ckpt = torch.load(pt_path, map_location="cpu")
        sd   = ckpt["model"]
        lw   = sd["level_weights"].float()                 # (12,) raw logits
        w    = F.softmax(lw, dim=0) * N_LEVELS             # softmax*N 가중치
        w_np = w.detach().cpu().numpy()
    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        continue

    rec = {"patient": patient_id, "modality": modality}
    for lvl in range(N_LEVELS):
        rec[f"L{lvl+1:02d}"] = w_np[lvl]
    records.append(rec)

df = pd.DataFrame(records)
level_cols = [f"L{i:02d}" for i in range(1, N_LEVELS+1)]
level_labels = [f"L{i}" for i in range(1, N_LEVELS+1)]

print(f"추출 완료: {len(df)} 케이스")
print(df["modality"].value_counts().to_string())

df.to_csv(os.path.join(OUT_DIR, "gating_weights_raw.csv"), index=False)
print("gating_weights_raw.csv 저장 완료")

# ================================================================
# Step 2: 레벨별 통계 요약 저장
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
stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(OUT_DIR, "gating_stats_summary.csv"), index=False)
print("gating_stats_summary.csv 저장 완료")

# ================================================================
# Step 3: Kruskal-Wallis 검정 (모달리티 간)
# ================================================================
kw_rows = []
for lvl in level_cols:
    groups = [df[df["modality"] == m][lvl].values for m in MODALITIES]
    h_stat, p_val = stats.kruskal(*groups)
    kw_rows.append({"level": lvl, "H_stat": round(h_stat, 4), "p_value": round(p_val, 6),
                    "significant": "YES" if p_val < 0.05 else "NO"})
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

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, all_mean, yerr=all_std, capsize=4,
              color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.8,
              error_kw={"ecolor": "#333333", "elinewidth": 1.2})
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Uniform (1.0)")
ax.set_xticks(x)
ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Gating Weight (softmax × N)")
ax.set_title("Overall Level Gating Weights — All 192 Cases (mean ± std)")
ax.legend()
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_overall_bar.png"), dpi=150)
plt.close()
print("fig1_overall_bar.png 저장 완료")

# ================================================================
# Fig 2: 전체 박스플롯
# ================================================================
fig, ax = plt.subplots(figsize=(11, 5))
bp = ax.boxplot(
    [df[lvl].values for lvl in level_cols],
    labels=level_labels, patch_artist=True,
    medianprops={"color": "red", "linewidth": 1.5},
    whiskerprops={"linewidth": 1.2},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
)
for patch in bp["boxes"]:
    patch.set(facecolor="#4C72B0", alpha=0.6)
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Uniform (1.0)")
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Gating Weight")
ax.set_title("Level Gating Weight Distribution — All 192 Cases")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_overall_boxplot.png"), dpi=150)
plt.close()
print("fig2_overall_boxplot.png 저장 완료")

# ================================================================
# Fig 3: 모달리티별 평균 프로파일 (선 그래프)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
for mod in MODALITIES:
    sub  = df[df["modality"] == mod]
    mean = sub[level_cols].mean().values
    std  = sub[level_cols].std().values
    ax.plot(x, mean, marker="o", markersize=5, linewidth=2,
            color=COLORS[mod], label=f"{mod} (n={len(sub)})")
    ax.fill_between(x, mean - std, mean + std,
                    color=COLORS[mod], alpha=0.12)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="Uniform (1.0)")
ax.set_xticks(x)
ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Gating Weight (softmax × N)")
ax.set_title("Gating Weight Profile by Modality (mean ± std shading)")
ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_modality_profile.png"), dpi=150)
plt.close()
print("fig3_modality_profile.png 저장 완료")

# ================================================================
# Fig 4: 히트맵 (케이스 × 레벨, 모달리티별 정렬)
# ================================================================
df_sorted = df.sort_values(["modality", "patient"])
heatmap_data = df_sorted[level_cols].values  # (192, 12)

fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn",
               vmin=0.5, vmax=1.6)
plt.colorbar(im, ax=ax, label="Gating Weight")

# 모달리티 경계선 + 레이블
mod_counts = df_sorted["modality"].value_counts()[MODALITIES].values
boundaries = np.cumsum(mod_counts)
y_offset = 0
for mod, cnt in zip(MODALITIES, mod_counts):
    ax.axhline(y=y_offset - 0.5, color="white", linewidth=2)
    ax.text(-0.8, y_offset + cnt / 2, mod, ha="right", va="center",
            fontsize=11, fontweight="bold", color=COLORS[mod])
    y_offset += cnt

ax.set_xticks(np.arange(N_LEVELS))
ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level (Low → High Frequency)")
ax.set_ylabel("Cases (grouped by modality)")
ax.set_title("Gating Weights Heatmap — All 192 Cases × 12 Levels")
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_heatmap.png"), dpi=150)
plt.close()
print("fig4_heatmap.png 저장 완료")

# ================================================================
# Fig 5: 레벨별 CV (변동계수) — 일관성 확인
# ================================================================
fig, ax = plt.subplots(figsize=(10, 4))
for mod in MODALITIES:
    sub = df[df["modality"] == mod]
    cv  = (sub[level_cols].std() / sub[level_cols].mean()).values
    ax.plot(x, cv, marker="s", markersize=4, linewidth=1.5,
            color=COLORS[mod], label=mod)
all_cv = (df[level_cols].std() / df[level_cols].mean()).values
ax.plot(x, all_cv, marker="D", markersize=5, linewidth=2,
        color="black", linestyle="--", label="ALL")
ax.set_xticks(x)
ax.set_xticklabels(level_labels)
ax.set_xlabel("Hash Grid Level")
ax.set_ylabel("CV (std / mean)")
ax.set_title("Coefficient of Variation per Level — Lower = More Consistent")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_cv.png"), dpi=150)
plt.close()
print("fig5_cv.png 저장 완료")

print(f"\n모든 분석 완료. 결과 폴더: {OUT_DIR}")

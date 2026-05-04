#!/usr/bin/env python3
"""
analyze_hsm_v3.py
Hard Slice Mining 분석 스크립트 v3 — 분석 1~10
변경 사항 (vs analyze_hsm_v2.py):
  - 분석 1~7: v2와 동일 (코드 재사용)
  - 분석 8 (신규): 에포크 내 배치 위치 — HM2 직접 증거 ①
      slice_avg_first_batch_pos 와 잔차의 Spearman
      → 음수이면 고잔차 슬라이스가 에포크 초반 배치에 집중됨 = HM2 작동
  - 분석 9 (신규): 슬라이스별 Loss 기여량 — HM2 직접 증거 ②
      slice_avg_loss_per_pixel 과 잔차의 Spearman
      → 양수이면 고잔차 슬라이스가 실제로 Loss를 더 많이 발생시킴
  - 분석 10 (신규): 에포크 전반/후반 Loss 기여 비율 — HM2 직접 증거 ③
      (loss_early / loss_late) 와 잔차의 Spearman
      → 양수이면 고잔차 슬라이스가 에포크 초반에 Loss를 더 많이 발생시킴

사용법: python analyze_hsm_v3.py
출력:   ./hsm_analysis_output_v3/ 아래 CSV + PNG
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from scipy import stats
from scipy.special import rel_entr

# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
RECON_ROOT  = "/dshome/ddualab/dongnyeok/NeSVoR/recon_HSM_hsm_O"
TEST_CSV    = "/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
OUT_DIR     = "./hsm_analysis_output_v3"
MODALITIES  = ["flair", "t1ce"]
EMA_ALPHA   = 0.99
SIM_ALPHAS  = [0.80, 0.90, 0.95, 0.99]
N_ITER      = 2000

os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/dshome/ddualab/dongnyeok/NanumGothic.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════
def load_case(patient_dir: str, patient_id: str, modality: str) -> Optional[dict]:
    stem         = f"{patient_id}_{modality}_4x5_hsm_O"
    analysis_dir = os.path.join(patient_dir, f"{stem}_slice_analysis")
    if not os.path.isdir(analysis_dir):
        return None

    required = [
        "slice_residuals_final.npy",
        "slice_sample_counts_main.npy",
        "slice_sample_counts_patch.npy",
        "slice_pixel_counts.npy",
        "residuals_history_iters.npy",
        "residuals_history_values.npy",
    ]
    # v3 신규 파일 (없으면 None으로 로드하고 분석 8~10에서 skip)
    optional_v3 = [
        "slice_avg_first_batch_pos.npy",
        "slice_avg_loss_per_pixel.npy",
        "slice_loss_early_sum.npy",
        "slice_loss_late_sum.npy",
    ]

    data = {}
    for fname in required:
        fpath = os.path.join(analysis_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] 파일 없음: {fpath}")
            return None
        data[fname.replace(".npy", "")] = np.load(fpath)

    for fname in optional_v3:
        fpath = os.path.join(analysis_dir, fname)
        key   = fname.replace(".npy", "")
        data[key] = np.load(fpath) if os.path.exists(fpath) else None

    data["patient"]  = patient_id
    data["modality"] = modality
    data["full_id"]  = os.path.basename(patient_dir)
    return data


def load_all_cases() -> List[dict]:
    df_ids   = pd.read_csv(TEST_CSV)
    full_ids = df_ids.iloc[:, 0].str.strip().tolist()
    cases    = []
    for full_id in full_ids:
        patient_id  = full_id.split("_")[-1]
        patient_dir = os.path.join(RECON_ROOT, full_id)
        for mod in MODALITIES:
            c = load_case(patient_dir, patient_id, mod)
            if c is not None:
                cases.append(c)
            else:
                print(f"  [SKIP] {full_id} / {mod}")
    print(f"\n총 {len(cases)}개 케이스 로드 완료\n")
    return cases


# ══════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════
def correct_counts_by_pixel(
    raw_counts: np.ndarray,
    pixel_counts: np.ndarray,
) -> np.ndarray:
    pix = pixel_counts.astype(np.float64)
    pix = np.where(pix == 0, 1.0, pix)
    return raw_counts.astype(np.float64) / pix


def _has_v3_data(c: dict) -> bool:
    """분석 8~10에 필요한 파일이 모두 있는지 확인."""
    return all(
        c.get(k) is not None
        for k in [
            "slice_avg_first_batch_pos",
            "slice_avg_loss_per_pixel",
            "slice_loss_early_sum",
            "slice_loss_late_sum",
        ]
    )


# ══════════════════════════════════════════════════════════════
# 분석 1~7: v2와 동일 — 함수 그대로 재사용
# ══════════════════════════════════════════════════════════════
def analysis1_residual_distribution(cases):
    print("=== 분석 1: 잔차 분포 & CV ===")
    rows = []
    for c in cases:
        r = c["slice_residuals_final"]
        rows.append({
            "patient":  c["patient"],
            "modality": c["modality"],
            "n_slices": len(r),
            "mean":     r.mean(),
            "std":      r.std(),
            "cv":       r.std() / (r.mean() + 1e-12),
            "min":      r.min(),
            "max":      r.max(),
            "p25":      np.percentile(r, 25),
            "p75":      np.percentile(r, 75),
            "iqr":      np.percentile(r, 75) - np.percentile(r, 25),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis1_residual_stats.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["cv"]
        ax.hist(sub, bins=20, edgecolor="white",
                color="steelblue" if mod == "flair" else "tomato")
        ax.axvline(sub.mean(), color="black", linestyle="--",
                   label=f"mean={sub.mean():.3f}")
        ax.set_title(f"CV Distribution — {mod.upper()}")
        ax.set_xlabel("Coefficient of Variation (std/mean)")
        ax.set_ylabel("# cases")
        ax.legend()
    fig.suptitle("Analysis 1: Residual CV per Case")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis1_cv_histogram.png"), dpi=150)
    plt.close()

    print(f"  CV 평균: flair={df[df.modality=='flair']['cv'].mean():.4f}, "
          f"t1ce={df[df.modality=='t1ce']['cv'].mean():.4f}")
    return df


def analysis2_sampling_bias(cases):
    print("=== 분석 2: 샘플링 편향 ===")
    rows = []
    for c in cases:
        r       = c["slice_residuals_final"].astype(np.float64)
        cm      = c["slice_sample_counts_main"].astype(np.float64)
        cp      = c["slice_sample_counts_patch"].astype(np.float64)
        pix     = c["slice_pixel_counts"].astype(np.float64)
        cm_corr = correct_counts_by_pixel(cm, pix)

        eps      = 1e-12
        r_prob   = r  / (r.sum()  + eps)
        cm_prob  = cm / (cm.sum() + eps)
        cp_prob  = cp / (cp.sum() + eps)
        n        = len(r)
        uniform  = np.ones(n) / n

        rho_main_raw,  _ = stats.spearmanr(r, cm)
        rho_main_corr, _ = stats.spearmanr(r, cm_corr)
        rho_patch,     _ = stats.spearmanr(r, cp)

        kl_main  = np.sum(rel_entr(r_prob + eps, cm_prob + eps))
        kl_patch = np.sum(rel_entr(r_prob + eps, cp_prob + eps))
        kl_unif  = np.sum(rel_entr(r_prob + eps, uniform + eps))

        rows.append({
            "patient":            c["patient"],
            "modality":           c["modality"],
            "spearman_main_raw":  rho_main_raw,
            "spearman_main_corr": rho_main_corr,
            "spearman_patch":     rho_patch,
            "kl_main":            kl_main,
            "kl_patch":           kl_patch,
            "kl_uniform_ref":     kl_unif,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis2_sampling_bias.csv"), index=False)

    # 그림 2-C: raw vs corrected 비교 바 차트
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bar_labels = ["main_raw", "main_corr", "patch"]
    for ax, mod in zip(axes, MODALITIES):
        sub   = df[df["modality"] == mod]
        means = [sub["spearman_main_raw"].mean(),
                 sub["spearman_main_corr"].mean(),
                 sub["spearman_patch"].mean()]
        sems  = [sub["spearman_main_raw"].sem(),
                 sub["spearman_main_corr"].sem(),
                 sub["spearman_patch"].sem()]
        colors = ["#e74c3c", "#e67e22", "#2ecc71"]
        bars   = ax.bar(bar_labels, means, yerr=sems, color=colors,
                        capsize=5, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-0.5, 0.8)
        ax.set_ylabel("Mean Spearman correlation")
        ax.set_title(f"{mod.upper()}")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.02 if val >= 0 else -0.05),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Analysis 2C: Spearman Mean Comparison\n"
                 "raw vs pixel-corrected vs patch  (error bar = SEM)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis2c_spearman_bar.png"), dpi=150)
    plt.close()

    print(f"  Spearman(main_raw)  평균: "
          f"flair={df[df.modality=='flair']['spearman_main_raw'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['spearman_main_raw'].mean():.3f}")
    print(f"  Spearman(main_corr) 평균: "
          f"flair={df[df.modality=='flair']['spearman_main_corr'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['spearman_main_corr'].mean():.3f}")
    print(f"  Spearman(patch)     평균: "
          f"flair={df[df.modality=='flair']['spearman_patch'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['spearman_patch'].mean():.3f}")
    return df


def analysis3_ema_convergence(cases):
    print("=== 분석 3: EMA 수렴 속도 ===")
    rows = []
    for c in cases:
        hist_iters  = c["residuals_history_iters"]
        hist_values = c["residuals_history_values"]
        n_slices    = hist_values.shape[1]
        var_series  = hist_values.var(axis=1)
        cm          = c["slice_sample_counts_main"].astype(np.float64)
        pix         = c["slice_pixel_counts"].astype(np.float64)
        cm_corr     = correct_counts_by_pixel(cm, pix)
        avg_updates = cm_corr.mean()
        init_retention = EMA_ALPHA ** avg_updates
        rows.append({
            "patient":              c["patient"],
            "modality":             c["modality"],
            "n_slices":             n_slices,
            "avg_update_per_slice": avg_updates,
            "init_retention_0.99":  init_retention,
            "var_initial":          var_series[0],
            "var_final":            var_series[-1],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis3_ema_convergence.csv"), index=False)
    print(f"  초기값 잔존율(alpha=0.99) 평균: "
          f"flair={df[df.modality=='flair']['init_retention_0.99'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['init_retention_0.99'].mean():.3f}")
    print(f"  슬라이스당 평균 업데이트 횟수: "
          f"flair={df[df.modality=='flair']['avg_update_per_slice'].mean():.1f}, "
          f"t1ce={df[df.modality=='t1ce']['avg_update_per_slice'].mean():.1f}")
    return df


def analysis5_hard_easy_slices(cases):
    print("=== 분석 5: Hard/Easy 슬라이스 특성 ===")
    rows = []
    for c in cases:
        r   = c["slice_residuals_final"]
        pix = c["slice_pixel_counts"].astype(np.float64)
        n   = len(r)
        top_idx = np.argsort(r)[-max(1, n // 5):]
        bot_idx = np.argsort(r)[:max(1, n // 5)]
        rows.append({
            "patient":               c["patient"],
            "modality":              c["modality"],
            "hard_mean_residual":    r[top_idx].mean(),
            "easy_mean_residual":    r[bot_idx].mean(),
            "hard_easy_ratio":       r[top_idx].mean() / (r[bot_idx].mean() + 1e-12),
            "hard_mean_pixels":      pix[top_idx].mean(),
            "easy_mean_pixels":      pix[bot_idx].mean(),
            "pixel_ratio_hard_easy": pix[top_idx].mean() / (pix[bot_idx].mean() + 1e-12),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis5_hard_easy.csv"), index=False)
    print(f"  Hard/Easy 잔차 비율 평균: "
          f"flair={df[df.modality=='flair']['hard_easy_ratio'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['hard_easy_ratio'].mean():.3f}")
    return df


def analysis6_modality_comparison(df1, df2, df3, df5):
    print("=== 분석 6: 모달리티 간 비교 ===")
    metrics = {
        "CV (잔차 변동계수)":              (df1, "cv"),
        "Spearman main (raw)":             (df2, "spearman_main_raw"),
        "Spearman main (pixel-corrected)": (df2, "spearman_main_corr"),
        "Spearman patch":                  (df2, "spearman_patch"),
        "Init Retention (alpha=0.99)":     (df3, "init_retention_0.99"),
        "Hard/Easy 잔차 비율":             (df5, "hard_easy_ratio"),
    }
    summary_rows = []
    for metric_name, (df, col) in metrics.items():
        for mod in MODALITIES:
            sub = df[df["modality"] == mod][col]
            summary_rows.append({
                "metric":   metric_name,
                "modality": mod,
                "mean":     sub.mean(),
                "std":      sub.std(),
                "median":   sub.median(),
            })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUT_DIR, "analysis6_modality_summary.csv"), index=False)
    print("  모달리티 비교 요약:")
    print(df_sum.to_string(index=False))


def analysis7_pixel_bias(cases):
    print("=== 분석 7: 슬라이스 크기 편향 검증 ===")
    rows = []
    for c in cases:
        r       = c["slice_residuals_final"].astype(np.float64)
        cm      = c["slice_sample_counts_main"].astype(np.float64)
        pix     = c["slice_pixel_counts"].astype(np.float64)
        cm_corr = correct_counts_by_pixel(cm, pix)
        rho_pix_cm_raw,  _ = stats.spearmanr(pix, cm)
        rho_pix_resid,   _ = stats.spearmanr(pix, r)
        rho_pix_cm_corr, _ = stats.spearmanr(pix, cm_corr)
        rows.append({
            "patient":         c["patient"],
            "modality":        c["modality"],
            "rho_pix_cm_raw":  rho_pix_cm_raw,
            "rho_pix_resid":   rho_pix_resid,
            "rho_pix_cm_corr": rho_pix_cm_corr,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis7_pixel_bias.csv"), index=False)
    print(f"  ρ(pixel, cm_raw)  평균: "
          f"flair={df[df.modality=='flair']['rho_pix_cm_raw'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['rho_pix_cm_raw'].mean():.3f}")
    print(f"  ρ(pixel, residual) 평균: "
          f"flair={df[df.modality=='flair']['rho_pix_resid'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['rho_pix_resid'].mean():.3f}")
    print(f"  ρ(pixel, cm_corr) 평균: "
          f"flair={df[df.modality=='flair']['rho_pix_cm_corr'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['rho_pix_cm_corr'].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 8 (신규) — 에포크 내 배치 위치 (HM2 직접 증거 ①)
# ══════════════════════════════════════════════════════════════
def analysis8_first_batch_pos(cases: List[dict]):
    """
    HM2가 작동하면 고잔차 슬라이스가 에포크 앞쪽 배치에 집중된다.
    slice_avg_first_batch_pos (슬라이스별 평균 첫 등장 배치 번호) 와
    slice_residuals_final 의 Spearman 을 계산.

    기대값: 음수 (잔차 높을수록 작은 배치 번호, 즉 에포크 앞쪽)
    HM2 가 없다면(균등 셔플): ≈ 0
    """
    print("=== 분석 8: 에포크 내 배치 위치 (HM2 직접 증거 ①) ===")

    v3_cases = [c for c in cases if _has_v3_data(c)]
    if not v3_cases:
        print("  [SKIP] slice_avg_first_batch_pos.npy 가 없습니다. "
              "train.py 수정 후 재학습 필요.")
        return None

    rows = []
    for c in v3_cases:
        r   = c["slice_residuals_final"].astype(np.float64)
        pos = c["slice_avg_first_batch_pos"].astype(np.float64)

        # pos 가 0인 슬라이스(한 번도 등장 안 한 경우)는 제외
        valid = pos > 0
        if valid.sum() < 3:
            continue

        rho, pval = stats.spearmanr(r[valid], pos[valid])
        rows.append({
            "patient":          c["patient"],
            "modality":         c["modality"],
            "spearman_pos":     rho,
            "pval":             pval,
            "n_valid_slices":   int(valid.sum()),
            "mean_pos":         pos[valid].mean(),
            "std_pos":          pos[valid].std(),
        })

    if not rows:
        print("  [WARN] 유효한 케이스 없음.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis8_first_batch_pos.csv"), index=False)

    # 그림 8-A: Spearman 분포 히스토그램
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["spearman_pos"]
        if sub.empty:
            continue
        color = "steelblue" if mod == "flair" else "tomato"
        ax.hist(sub, bins=20, edgecolor="white", color=color)
        ax.axvline(sub.mean(), color="black", linestyle="--",
                   label=f"mean={sub.mean():.3f}")
        ax.axvline(0, color="gray", linestyle=":", lw=1.5)
        ax.set_xlabel("Spearman(residual, avg_first_batch_pos)")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()}")
        ax.legend()
    fig.suptitle("Analysis 8: Spearman(residual, first_batch_pos)\n"
                 "음수 = 고잔차 슬라이스가 에포크 앞쪽 배치에 집중 → HM2 작동")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis8a_first_batch_pos_hist.png"), dpi=150)
    plt.close()

    # 그림 8-B: 대표 케이스 scatter (residual vs first_batch_pos)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df = df[df["modality"] == mod]
        if sub_df.empty:
            continue
        # 절댓값 최대 Spearman 케이스를 대표로 선택
        rep_row  = sub_df.iloc[sub_df["spearman_pos"].abs().argsort().iloc[-1]]
        rep_case = next(c for c in v3_cases
                        if c["patient"] == rep_row["patient"]
                        and c["modality"] == mod)
        r   = rep_case["slice_residuals_final"].astype(np.float64)
        pos = rep_case["slice_avg_first_batch_pos"].astype(np.float64)
        valid = pos > 0
        color = "steelblue" if mod == "flair" else "tomato"
        ax.scatter(r[valid], pos[valid], alpha=0.5, s=20, color=color)
        ax.set_xlabel("Slice residual (EMA)")
        ax.set_ylabel("Avg first batch position in epoch")
        ax.set_title(f"{mod.upper()} — Patient {rep_row['patient']}\n"
                     f"ρ={rep_row['spearman_pos']:.3f}, p={rep_row['pval']:.3f}")
    fig.suptitle("Analysis 8B: Residual vs First Batch Position\n"
                 "(왼쪽 아래로 음의 상관 → HM2가 고잔차 슬라이스를 앞쪽 배치에 배치)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis8b_first_batch_pos_scatter.png"), dpi=150)
    plt.close()

    # 그림 8-C: flair / t1ce Spearman 바 비교
    fig, ax = plt.subplots(figsize=(6, 5))
    mods    = MODALITIES
    means   = [df[df["modality"] == m]["spearman_pos"].mean() for m in mods]
    sems    = [df[df["modality"] == m]["spearman_pos"].sem()  for m in mods]
    colors  = ["steelblue", "tomato"]
    bars    = ax.bar(mods, means, yerr=sems, color=colors, capsize=6, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean Spearman(residual, first_batch_pos)")
    ax.set_title("Analysis 8C: HM2 First-Batch-Position Effect\n"
                 "음수 = HM2 작동 (고잔차 → 앞쪽 배치)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.01 if val >= 0 else -0.03),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis8c_first_batch_pos_bar.png"), dpi=150)
    plt.close()

    for mod in MODALITIES:
        sub = df[df["modality"] == mod]
        if not sub.empty:
            print(f"  Spearman(residual, first_batch_pos) {mod}: "
                  f"mean={sub['spearman_pos'].mean():.3f}, "
                  f"median={sub['spearman_pos'].median():.3f}, "
                  f"n={len(sub)}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 9 (신규) — 슬라이스별 Loss 기여량 (HM2 직접 증거 ②)
# ══════════════════════════════════════════════════════════════
def analysis9_loss_contribution(cases: List[dict]):
    """
    slice_avg_loss_per_pixel (슬라이스별 픽셀당 평균 Loss 기여) 과
    slice_residuals_final 의 Spearman 을 계산.

    기대값: 양수 (고잔차 슬라이스가 실제로 Loss를 더 많이 발생시킴)
    추가로 patch Spearman 과 비교하여 HM2 vs patch HSM 효과를 나란히 제시.
    """
    print("=== 분석 9: 슬라이스별 Loss 기여량 (HM2 직접 증거 ②) ===")

    v3_cases = [c for c in cases if _has_v3_data(c)]
    if not v3_cases:
        print("  [SKIP] slice_avg_loss_per_pixel.npy 가 없습니다.")
        return None

    rows = []
    for c in v3_cases:
        r    = c["slice_residuals_final"].astype(np.float64)
        loss = c["slice_avg_loss_per_pixel"].astype(np.float64)
        cp   = c["slice_sample_counts_patch"].astype(np.float64)

        # loss 가 0인 슬라이스 제외
        valid = loss > 0
        if valid.sum() < 3:
            continue

        rho_loss,  pval_loss  = stats.spearmanr(r[valid], loss[valid])
        rho_patch, _          = stats.spearmanr(r, cp)

        rows.append({
            "patient":          c["patient"],
            "modality":         c["modality"],
            "spearman_loss":    rho_loss,
            "pval_loss":        pval_loss,
            "spearman_patch":   rho_patch,
            "n_valid_slices":   int(valid.sum()),
        })

    if not rows:
        print("  [WARN] 유효한 케이스 없음.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis9_loss_contribution.csv"), index=False)

    # 그림 9-A: Spearman(loss) 분포
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["spearman_loss"]
        if sub.empty:
            continue
        color = "steelblue" if mod == "flair" else "tomato"
        ax.hist(sub, bins=20, edgecolor="white", color=color)
        ax.axvline(sub.mean(), color="black", linestyle="--",
                   label=f"mean={sub.mean():.3f}")
        ax.axvline(0, color="gray", linestyle=":", lw=1.5)
        ax.set_xlabel("Spearman(residual, avg_loss_per_pixel)")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()}")
        ax.legend()
    fig.suptitle("Analysis 9: Spearman(residual, avg_loss_per_pixel)\n"
                 "양수 = 고잔차 슬라이스가 실제로 Loss 더 많이 발생 → HM2 효과")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis9a_loss_contribution_hist.png"), dpi=150)
    plt.close()

    # 그림 9-B: loss Spearman vs patch Spearman scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]
        if sub.empty:
            continue
        color = "steelblue" if mod == "flair" else "tomato"
        ax.scatter(sub["spearman_loss"], sub["spearman_patch"],
                   alpha=0.6, s=30, color=color)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("Spearman(residual, avg_loss_per_pixel)  [HM2]")
        ax.set_ylabel("Spearman(residual, patch_counts)  [patch HSM]")
        ax.set_title(f"{mod.upper()}\n"
                     f"loss mean={sub['spearman_loss'].mean():.3f}, "
                     f"patch mean={sub['spearman_patch'].mean():.3f}")
    fig.suptitle("Analysis 9B: HM2 (Loss) vs Patch HSM Spearman\n"
                 "1사분면 = 두 메커니즘 모두 고잔차 슬라이스를 더 많이 학습")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis9b_loss_vs_patch_scatter.png"), dpi=150)
    plt.close()

    # 그림 9-C: loss vs patch Spearman 바 비교
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bar_labels = ["HM2\n(loss/pixel)", "patch HSM\n(patch counts)"]
    for ax, mod in zip(axes, MODALITIES):
        sub   = df[df["modality"] == mod]
        means = [sub["spearman_loss"].mean(), sub["spearman_patch"].mean()]
        sems  = [sub["spearman_loss"].sem(),  sub["spearman_patch"].sem()]
        colors_bar = ["#9b59b6", "#2ecc71"]
        bars  = ax.bar(bar_labels, means, yerr=sems, color=colors_bar,
                       capsize=5, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-0.3, 0.8)
        ax.set_ylabel("Mean Spearman correlation")
        ax.set_title(f"{mod.upper()}")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.02 if val >= 0 else -0.05),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Analysis 9C: HM2 vs Patch HSM — Spearman Comparison\n"
                 "(error bar = SEM)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis9c_hm2_vs_patch_bar.png"), dpi=150)
    plt.close()

    for mod in MODALITIES:
        sub = df[df["modality"] == mod]
        if not sub.empty:
            print(f"  Spearman(residual, loss_per_pixel) {mod}: "
                  f"mean={sub['spearman_loss'].mean():.3f}, "
                  f"median={sub['spearman_loss'].median():.3f}")
            print(f"  Spearman(residual, patch_counts)   {mod}: "
                  f"mean={sub['spearman_patch'].mean():.3f}, "
                  f"median={sub['spearman_patch'].median():.3f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 10 (신규) — 에포크 전반/후반 Loss 비율 (HM2 직접 증거 ③)
# ══════════════════════════════════════════════════════════════
def analysis10_early_late_loss(cases: List[dict]):
    """
    slice_loss_early_sum / slice_loss_late_sum 비율(early_late_ratio) 과
    slice_residuals_final 의 Spearman 을 계산.

    기대값: 양수 (고잔차 슬라이스일수록 에포크 초반 Loss 비중이 큼)
    HM2 가 없다면: ≈ 0 (early/late 비율이 잔차와 무관)
    """
    print("=== 분석 10: 에포크 전반/후반 Loss 비율 (HM2 직접 증거 ③) ===")

    v3_cases = [c for c in cases if _has_v3_data(c)]
    if not v3_cases:
        print("  [SKIP] slice_loss_early_sum.npy 가 없습니다.")
        return None

    rows = []
    for c in v3_cases:
        r     = c["slice_residuals_final"].astype(np.float64)
        early = c["slice_loss_early_sum"].astype(np.float64)
        late  = c["slice_loss_late_sum"].astype(np.float64)

        # late 가 0인 슬라이스 제외 (에포크 후반에 한 번도 등장 안 한 경우)
        valid = late > 0
        if valid.sum() < 3:
            continue

        ratio = early[valid] / (late[valid] + 1e-12)
        rho, pval = stats.spearmanr(r[valid], ratio)

        rows.append({
            "patient":              c["patient"],
            "modality":             c["modality"],
            "spearman_early_late":  rho,
            "pval":                 pval,
            "mean_ratio":           ratio.mean(),
            "std_ratio":            ratio.std(),
            "n_valid_slices":       int(valid.sum()),
        })

    if not rows:
        print("  [WARN] 유효한 케이스 없음.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis10_early_late_loss.csv"), index=False)

    # 그림 10-A: Spearman 분포 히스토그램
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["spearman_early_late"]
        if sub.empty:
            continue
        color = "steelblue" if mod == "flair" else "tomato"
        ax.hist(sub, bins=20, edgecolor="white", color=color)
        ax.axvline(sub.mean(), color="black", linestyle="--",
                   label=f"mean={sub.mean():.3f}")
        ax.axvline(0, color="gray", linestyle=":", lw=1.5)
        ax.set_xlabel("Spearman(residual, early/late loss ratio)")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()}")
        ax.legend()
    fig.suptitle("Analysis 10: Spearman(residual, early/late loss ratio)\n"
                 "양수 = 고잔차 슬라이스가 에포크 초반에 Loss 더 많이 발생 → HM2 효과")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis10a_early_late_hist.png"), dpi=150)
    plt.close()

    # 그림 10-B: 대표 케이스 — residual vs early/late ratio scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df = df[df["modality"] == mod]
        if sub_df.empty:
            continue
        rep_row  = sub_df.iloc[sub_df["spearman_early_late"].abs().argsort().iloc[-1]]
        rep_case = next(c for c in v3_cases
                        if c["patient"] == rep_row["patient"]
                        and c["modality"] == mod)
        r     = rep_case["slice_residuals_final"].astype(np.float64)
        early = rep_case["slice_loss_early_sum"].astype(np.float64)
        late  = rep_case["slice_loss_late_sum"].astype(np.float64)
        valid = late > 0
        ratio = early[valid] / (late[valid] + 1e-12)
        color = "steelblue" if mod == "flair" else "tomato"
        ax.scatter(r[valid], ratio, alpha=0.5, s=20, color=color)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="ratio=1 (동등)")
        ax.set_xlabel("Slice residual (EMA)")
        ax.set_ylabel("early_loss / late_loss ratio")
        ax.set_title(f"{mod.upper()} — Patient {rep_row['patient']}\n"
                     f"ρ={rep_row['spearman_early_late']:.3f}, "
                     f"p={rep_row['pval']:.3f}")
        ax.legend()
    fig.suptitle("Analysis 10B: Residual vs Early/Late Loss Ratio\n"
                 "(우측 위 = 고잔차 슬라이스가 초반 배치에서 Loss 더 많이 기여)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis10b_early_late_scatter.png"), dpi=150)
    plt.close()

    # 그림 10-C: 분석 8, 9, 10 Spearman 3종 종합 바 차트
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub10 = df[df["modality"] == mod]
        if sub10.empty:
            continue
        # 분석 8, 9 결과를 이미 저장된 CSV에서 불러와 함께 표시
        csv8 = os.path.join(OUT_DIR, "analysis8_first_batch_pos.csv")
        csv9 = os.path.join(OUT_DIR, "analysis9_loss_contribution.csv")
        bar_data = []
        bar_labels_c = []
        bar_colors_c = []
        if os.path.exists(csv8):
            df8 = pd.read_csv(csv8)
            sub8 = df8[df8["modality"] == mod]
            if not sub8.empty:
                bar_data.append((sub8["spearman_pos"].mean(),
                                 sub8["spearman_pos"].sem()))
                bar_labels_c.append("①\nfirst_batch_pos\n(음수 기대)")
                bar_colors_c.append("#3498db")
        if os.path.exists(csv9):
            df9 = pd.read_csv(csv9)
            sub9 = df9[df9["modality"] == mod]
            if not sub9.empty:
                bar_data.append((sub9["spearman_loss"].mean(),
                                 sub9["spearman_loss"].sem()))
                bar_labels_c.append("②\nloss_per_pixel\n(양수 기대)")
                bar_colors_c.append("#9b59b6")
        bar_data.append((sub10["spearman_early_late"].mean(),
                         sub10["spearman_early_late"].sem()))
        bar_labels_c.append("③\nearly/late ratio\n(양수 기대)")
        bar_colors_c.append("#e67e22")

        if bar_data:
            means_c = [d[0] for d in bar_data]
            sems_c  = [d[1] for d in bar_data]
            bars = ax.bar(bar_labels_c, means_c, yerr=sems_c,
                          color=bar_colors_c, capsize=5, edgecolor="white")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylim(-0.6, 0.8)
            ax.set_ylabel("Mean Spearman correlation")
            ax.set_title(f"{mod.upper()} — HM2 3종 증거")
            for bar, val in zip(bars, means_c):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + (0.02 if val >= 0 else -0.06),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Analysis 10C: HM2 Direct Evidence Summary\n"
                 "①음수 / ②③양수이면 HM2가 설계대로 동작")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis10c_hm2_summary_bar.png"), dpi=150)
    plt.close()

    for mod in MODALITIES:
        sub = df[df["modality"] == mod]
        if not sub.empty:
            print(f"  Spearman(residual, early/late ratio) {mod}: "
                  f"mean={sub['spearman_early_late'].mean():.3f}, "
                  f"median={sub['spearman_early_late'].median():.3f}")
    return df


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cases = load_all_cases()
    if not cases:
        print("로드된 케이스가 없습니다. 경로를 확인해 주세요.")
        exit(1)

    df1 = analysis1_residual_distribution(cases)
    df2 = analysis2_sampling_bias(cases)
    df3 = analysis3_ema_convergence(cases)
    df5 = analysis5_hard_easy_slices(cases)
    analysis6_modality_comparison(df1, df2, df3, df5)
    analysis7_pixel_bias(cases)

    # v3 신규 분석 (train.py 수정 후 재학습된 데이터에서만 실행)
    df8  = analysis8_first_batch_pos(cases)
    df9  = analysis9_loss_contribution(cases)
    df10 = analysis10_early_late_loss(cases)

    v3_count = sum(1 for c in cases if _has_v3_data(c))
    print(f"\n  [v3 데이터 보유 케이스: {v3_count} / {len(cases)}]")
    print(f"\n✅ 모든 분석 완료. 결과 저장 위치: {OUT_DIR}")

#!/usr/bin/env python3
"""
analyze_hsm_v2.py
Hard Slice Mining 분석 스크립트 v2 — 분석 1~7
변경 사항 (vs analyze_hsm.py):
  - 분석 2: slice_sample_counts_main 픽셀 수 편향 보정
            (픽셀 수 누적 카운트 → 슬라이스 픽셀 수로 나눈 등장 횟수 근사)
            보정 전/후 Spearman 비교 추가
  - 분석 3: EMA 업데이트 횟수를 픽셀 수 추정치 대신
            slice_sample_counts_main / slice_pixel_counts 로 정확하게 계산
  - 분석 7 (신규): 슬라이스 크기 편향 검증
            픽셀 수 vs 잔차 vs 샘플링 카운트의 삼각 관계 직접 시각화

사용법: python analyze_hsm_v2.py
출력:   ./hsm_analysis_output_v2/ 아래 CSV + PNG
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
# RECON_ROOT  = "/dshome/ddualab/dongnyeok/NeSVoR/recon_analysis"
RECON_ROOT  = "/dshome/ddualab/dongnyeok/NeSVoR/recon_HSM_hsm_O"
TEST_CSV    = "/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
OUT_DIR     = "./hsm_analysis_output_v2"
MODALITIES  = ["flair", "t1ce"]
EMA_ALPHA   = 0.99
SIM_ALPHAS  = [0.80, 0.90, 0.95, 0.99]
N_ITER      = 2000

os.makedirs(OUT_DIR, exist_ok=True)

font_path   = "/dshome/ddualab/dongnyeok/NanumGothic.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════
def load_case(patient_dir: str, patient_id: str, modality: str) -> Optional[dict]:
    # stem         = f"{patient_id}_{modality}_4x5_analysis"
    stem = f"{patient_id}_{modality}_4x5_hsm_O"
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
    data = {}
    for fname in required:
        fpath = os.path.join(analysis_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] 파일 없음: {fpath}")
            return None
        data[fname.replace(".npy", "")] = np.load(fpath)

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
# 유틸: 픽셀 수 편향 보정
# ══════════════════════════════════════════════════════════════
def correct_counts_by_pixel(
    raw_counts: np.ndarray,
    pixel_counts: np.ndarray,
) -> np.ndarray:
    """
    slice_sample_counts_main 은 픽셀 수를 누적한 값이다.
    슬라이스 i 의 "배치 등장 횟수 근사" = raw_counts[i] / pixel_counts[i]
    → 슬라이스 크기 편향을 제거한 상대적 방문 빈도.
    반환값은 float 배열 (절대 횟수가 아닌 비례값).
    """
    pix = pixel_counts.astype(np.float64)
    pix = np.where(pix == 0, 1.0, pix)          # 0 분모 방지
    return raw_counts.astype(np.float64) / pix


# ══════════════════════════════════════════════════════════════
# 분석 1 — 잔차 분포 & CV
# ══════════════════════════════════════════════════════════════
def analysis1_residual_distribution(cases: List[dict]):
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df   = df[df["modality"] == mod]
        med_cv   = sub_df["cv"].median()
        rep_row  = sub_df.iloc[(sub_df["cv"] - med_cv).abs().argsort().iloc[0]]
        rep_case = next(c for c in cases
                        if c["patient"] == rep_row["patient"]
                        and c["modality"] == mod)
        r = rep_case["slice_residuals_final"]
        ax.hist(r, bins=30, edgecolor="white",
                color="steelblue" if mod == "flair" else "tomato")
        ax.set_title(f"{mod.upper()} — Patient {rep_row['patient']}\n"
                     f"CV={rep_row['cv']:.3f} (median)")
        ax.set_xlabel("Slice Residual (EMA)")
        ax.set_ylabel("# slices")
    fig.suptitle("Analysis 1: Representative Residual Histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis1_residual_histogram_repr.png"), dpi=150)
    plt.close()

    print(f"  CV 평균: flair={df[df.modality=='flair']['cv'].mean():.4f}, "
          f"t1ce={df[df.modality=='t1ce']['cv'].mean():.4f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 2 — 샘플링 편향 (잔차 vs 실제 카운트) [v2: 픽셀 보정 추가]
# ══════════════════════════════════════════════════════════════
def analysis2_sampling_bias(cases: List[dict]):
    """
    v2 변경사항:
      - slice_sample_counts_main 은 픽셀 수 누적이므로,
        슬라이스 픽셀 수로 나눈 보정값(cm_corrected)도 함께 Spearman 계산.
      - Scatter: (Spearman_main_raw, Spearman_patch) 외에
                 (Spearman_main_corrected, Spearman_patch) 추가.
      - 해석: raw Spearman 이 음수더라도 corrected 가 양수라면
              샘플링 자체는 잔차 기반으로 동작하고 있으며
              음수의 원인이 픽셀 수 편향임을 확인할 수 있음.
    """
    print("=== 분석 2: 샘플링 편향 ===")
    rows = []
    for c in cases:
        r   = c["slice_residuals_final"].astype(np.float64)
        cm  = c["slice_sample_counts_main"].astype(np.float64)
        cp  = c["slice_sample_counts_patch"].astype(np.float64)
        pix = c["slice_pixel_counts"].astype(np.float64)

        # 픽셀 수 보정 카운트
        cm_corr = correct_counts_by_pixel(cm, pix)

        # 이론적 샘플링 확률 (잔차 기반)
        eps     = 1e-12
        r_prob  = r  / (r.sum()  + eps)
        cm_prob = cm / (cm.sum() + eps)
        cp_prob = cp / (cp.sum() + eps)
        n       = len(r)
        uniform = np.ones(n) / n

        # Spearman
        rho_main_raw,  _ = stats.spearmanr(r, cm)
        rho_main_corr, _ = stats.spearmanr(r, cm_corr)
        rho_patch,     _ = stats.spearmanr(r, cp)

        # KL-divergence
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

    # 그림 2-A: Scatter — raw main vs patch  (기존과 동일, 비교 기준)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]
        ax.scatter(sub["spearman_main_raw"], sub["spearman_patch"],
                   alpha=0.6, s=30, color="steelblue" if mod == "flair" else "tomato")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("Spearman(residual, main_counts_raw)")
        ax.set_ylabel("Spearman(residual, patch_counts)")
        ax.set_title(f"{mod.upper()}\nmain_raw mean={sub['spearman_main_raw'].mean():.3f}, "
                     f"patch mean={sub['spearman_patch'].mean():.3f}")
    fig.suptitle("Analysis 2A: Residual–SamplingCount Spearman\n"
                 "(main_counts = raw pixel accumulation)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis2a_spearman_scatter_raw.png"), dpi=150)
    plt.close()

    # 그림 2-B: Scatter — corrected main vs patch  (v2 신규)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]
        ax.scatter(sub["spearman_main_corr"], sub["spearman_patch"],
                   alpha=0.6, s=30, color="steelblue" if mod == "flair" else "tomato")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("Spearman(residual, main_counts_corrected)")
        ax.set_ylabel("Spearman(residual, patch_counts)")
        ax.set_title(f"{mod.upper()}\nmain_corr mean={sub['spearman_main_corr'].mean():.3f}, "
                     f"patch mean={sub['spearman_patch'].mean():.3f}")
    fig.suptitle("Analysis 2B: Residual–SamplingCount Spearman\n"
                 "(main_counts ÷ slice_pixel_counts = visit-frequency corrected)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis2b_spearman_scatter_corrected.png"), dpi=150)
    plt.close()

    # 그림 2-C: raw vs corrected 비교 바 차트
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bar_labels = ["main_raw", "main_corr", "patch"]
    for ax, mod in zip(axes, MODALITIES):
        sub    = df[df["modality"] == mod]
        means  = [
            sub["spearman_main_raw"].mean(),
            sub["spearman_main_corr"].mean(),
            sub["spearman_patch"].mean(),
        ]
        sems   = [
            sub["spearman_main_raw"].sem(),
            sub["spearman_main_corr"].sem(),
            sub["spearman_patch"].sem(),
        ]
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


# ══════════════════════════════════════════════════════════════
# 분석 3 — EMA 수렴 속도  [v2: 업데이트 횟수 계산 수정]
# ══════════════════════════════════════════════════════════════
def analysis3_ema_convergence(cases: List[dict]):
    """
    v2 변경사항:
      - 슬라이스당 평균 EMA 업데이트 횟수를
        (픽셀 수 기반 추정) → (slice_sample_counts_main / slice_pixel_counts) 로 수정.
        이유: train.py 에서 s_idx.unique() 기준으로 슬라이스별 1회 EMA 업데이트 수행.
             배치에 포함된 슬라이스 등장 횟수 = cm[i] / pix[i] 의 근사.
    """
    print("=== 분석 3: EMA 수렴 속도 ===")
    rows = []
    for c in cases:
        hist_iters  = c["residuals_history_iters"]
        hist_values = c["residuals_history_values"]
        n_slices    = hist_values.shape[1]
        var_series  = hist_values.var(axis=1)

        # v2: 슬라이스별 EMA 업데이트 횟수 = 픽셀 보정 카운트
        cm          = c["slice_sample_counts_main"].astype(np.float64)
        pix         = c["slice_pixel_counts"].astype(np.float64)
        cm_corr     = correct_counts_by_pixel(cm, pix)
        # 슬라이스당 평균 등장 횟수
        avg_updates = cm_corr.mean()

        # alpha=0.99 초기값 잔존율
        init_retention = EMA_ALPHA ** avg_updates

        rows.append({
            "patient":              c["patient"],
            "modality":             c["modality"],
            "n_slices":             n_slices,
            "avg_update_per_slice": avg_updates,
            "init_retention_0.99":  init_retention,
            "var_initial":          var_series[0],
            "var_final":            var_series[-1],
            "delta_var":            var_series[-1] - var_series[0],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis3_ema_convergence.csv"), index=False)

    # 그림 3-A: 슬라이스당 평균 업데이트 횟수
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]
        ax.hist(sub["avg_update_per_slice"], bins=20, edgecolor="white",
                color="steelblue" if mod == "flair" else "tomato")
        ax.axvline(sub["avg_update_per_slice"].mean(), color="black", linestyle="--",
                   label=f"mean={sub['avg_update_per_slice'].mean():.1f}")
        ax.set_xlabel("Avg # EMA updates per slice\n(= avg of cm_i / pix_i)")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()} — EMA Update Count per Slice (v2 corrected)")
        ax.legend()
    fig.suptitle("Analysis 3A: How many times was each slice updated?\n"
                 "(v2: pixel-count corrected estimate)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis3a_update_count.png"), dpi=150)
    plt.close()

    # 그림 3-B: 초기값 잔존율
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["init_retention_0.99"]
        ax.hist(sub, bins=20, edgecolor="white",
                color="steelblue" if mod == "flair" else "tomato")
        ax.axvline(sub.mean(), color="black", linestyle="--",
                   label=f"mean={sub.mean():.3f}")
        ax.axvline(0.5, color="red", linestyle=":", label="50% threshold")
        ax.set_xlabel("Initial value retention  alpha^k")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()} — Initial Retention (alpha=0.99)")
        ax.legend()
    fig.suptitle("Analysis 3B: alpha=0.99 초기값 잔존율\n"
                 "(0.5 이상 = EMA 가 실제 잔차보다 초기값에 더 가까움)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis3b_init_retention.png"), dpi=150)
    plt.close()

    # 그림 3-C: 대표 케이스 잔차 분산 시계열
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df   = df[df["modality"] == mod]
        med_n    = sub_df["n_slices"].median()
        rep_row  = sub_df.iloc[(sub_df["n_slices"] - med_n).abs().argsort().iloc[0]]
        rep_case = next(c for c in cases
                        if c["patient"] == rep_row["patient"]
                        and c["modality"] == mod)
        hist_iters  = rep_case["residuals_history_iters"]
        hist_values = rep_case["residuals_history_values"]
        var_series  = hist_values.var(axis=1)
        ax.plot(hist_iters, var_series, marker="o",
                color="steelblue" if mod == "flair" else "tomato")
        ax.set_xlabel("Training iter")
        ax.set_ylabel("Variance of slice residuals")
        ax.set_title(f"{mod.upper()} — Patient {rep_row['patient']}")
    fig.suptitle("Analysis 3C: Residual Variance over Training\n"
                 "(평탄 = 슬라이스 간 잔차 차이가 생기지 않음 → HSM 무효)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis3c_variance_timeseries.png"), dpi=150)
    plt.close()

    print(f"  초기값 잔존율(alpha=0.99) 평균: "
          f"flair={df[df.modality=='flair']['init_retention_0.99'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['init_retention_0.99'].mean():.3f}")
    print(f"  슬라이스당 평균 업데이트 횟수: "
          f"flair={df[df.modality=='flair']['avg_update_per_slice'].mean():.1f}, "
          f"t1ce={df[df.modality=='t1ce']['avg_update_per_slice'].mean():.1f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 4 — Alpha 하이퍼파라미터 시뮬레이션  (기존과 동일)
# ══════════════════════════════════════════════════════════════
def analysis4_alpha_simulation(cases: List[dict]):
    print("=== 분석 4: Alpha 시뮬레이션 ===")
    T_max = min(c["residuals_history_iters"].shape[0] for c in cases)
    alpha_var_curves = {a: [] for a in SIM_ALPHAS}

    for c in cases:
        hist_values = c["residuals_history_values"][:T_max]
        hist_iters  = c["residuals_history_iters"][:T_max]
        T, n_slices = hist_values.shape
        delta_iters = np.diff(hist_iters, prepend=0)

        signals    = np.zeros_like(hist_values)
        signals[0] = hist_values[0]
        for t in range(1, T):
            d    = delta_iters[t]
            aD   = EMA_ALPHA ** d
            denom = 1.0 - aD
            signals[t] = (
                hist_values[t] if denom < 1e-10
                else np.clip((hist_values[t] - aD * hist_values[t - 1]) / denom, 0, None)
            )

        for alpha in SIM_ALPHAS:
            ema       = signals[0].copy()
            var_curve = []
            for t in range(T):
                d  = delta_iters[t]
                aD = alpha ** d
                ema = aD * ema + (1.0 - aD) * signals[t]
                var_curve.append(ema.var())
            alpha_var_curves[alpha].append(var_curve)

    mean_curves = {
        a: np.array(alpha_var_curves[a]).mean(axis=0) for a in SIM_ALPHAS
    }
    ref_iters = cases[0]["residuals_history_iters"][:T_max]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    for alpha, color in zip(SIM_ALPHAS, colors):
        lw = 2.5 if alpha == EMA_ALPHA else 1.5
        ls = "--" if alpha == EMA_ALPHA else "-"
        ax.plot(ref_iters, mean_curves[alpha],
                label=f"alpha={alpha}" + (" (현재)" if alpha == EMA_ALPHA else ""),
                color=color, lw=lw, ls=ls)
    ax.set_xlabel("Training iter")
    ax.set_ylabel("Avg variance of slice residuals (96 cases)")
    ax.set_title("Analysis 4: Alpha Simulation — Residual Variance over Training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis4_alpha_simulation.png"), dpi=150)
    plt.close()

    df_out = pd.DataFrame({"iter": ref_iters})
    for alpha in SIM_ALPHAS:
        df_out[f"var_alpha_{alpha}"] = mean_curves[alpha]
    df_out.to_csv(os.path.join(OUT_DIR, "analysis4_alpha_simulation.csv"), index=False)

    k_arr  = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(10, 5))
    for alpha, color in zip(SIM_ALPHAS, colors):
        lw = 2.5 if alpha == EMA_ALPHA else 1.5
        ls = "--" if alpha == EMA_ALPHA else "-"
        ax.plot(k_arr, alpha ** k_arr,
                label=f"alpha={alpha}" + (" (현재)" if alpha == EMA_ALPHA else ""),
                color=color, lw=lw, ls=ls)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", label="50% retention")
    ax.set_xlabel("# EMA updates per slice (k)")
    ax.set_ylabel("Initial value retention  alpha^k")
    ax.set_title("Analysis 4B: Theoretical Initial Retention by Alpha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis4b_retention_curve.png"), dpi=150)
    plt.close()

    print("  alpha 시뮬레이션 완료. 최종 분산 비교:")
    for alpha in SIM_ALPHAS:
        print(f"    alpha={alpha}: final_var_mean={mean_curves[alpha][-1]:.6f}")


# ══════════════════════════════════════════════════════════════
# 분석 5 — Hard / Easy 슬라이스 특성  (기존과 동일)
# ══════════════════════════════════════════════════════════════
def analysis5_hard_easy_slices(cases: List[dict]):
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(
        axes,
        ["hard_easy_ratio", "pixel_ratio_hard_easy"],
        ["Hard/Easy Residual Ratio", "Hard/Easy Pixel Count Ratio"],
    ):
        for mod, color in zip(MODALITIES, ["steelblue", "tomato"]):
            sub = df[df["modality"] == mod][col]
            ax.hist(sub, bins=20, alpha=0.6, label=mod,
                    edgecolor="white", color=color)
        ax.axvline(1.0, color="black", ls="--", lw=1, label="ratio=1")
        ax.set_xlabel(col)
        ax.set_ylabel("# cases")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Analysis 5: Hard vs Easy Slice Characteristics")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis5_hard_easy.png"), dpi=150)
    plt.close()

    print(f"  Hard/Easy 잔차 비율 평균: "
          f"flair={df[df.modality=='flair']['hard_easy_ratio'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['hard_easy_ratio'].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 6 — 모달리티 간 비교 종합
# ══════════════════════════════════════════════════════════════
def analysis6_modality_comparison(df1, df2, df3, df5):
    print("=== 분석 6: 모달리티 간 비교 ===")
    metrics = {
        "CV (잔차 변동계수)":             (df1, "cv"),
        "Spearman main (raw)":            (df2, "spearman_main_raw"),
        "Spearman main (pixel-corrected)":(df2, "spearman_main_corr"),
        "Spearman patch":                 (df2, "spearman_patch"),
        "Init Retention (alpha=0.99)":    (df3, "init_retention_0.99"),
        "Hard/Easy 잔차 비율":            (df5, "hard_easy_ratio"),
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

    spearman_metrics = {
        k: v for k, v in metrics.items() if "Spearman" in k
    }
    other_metrics = {
        k: v for k, v in metrics.items() if "Spearman" not in k
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {"flair": "steelblue", "t1ce": "tomato"}
    all_metrics = list(metrics.items())
    for ax, (metric_name, (df, col)) in zip(axes.flat, all_metrics):
        data_by_mod = [df[df["modality"] == mod][col].values for mod in MODALITIES]
        bp = ax.boxplot(data_by_mod, patch_artist=True, widths=0.5)
        for patch, mod in zip(bp["boxes"], MODALITIES):
            patch.set_facecolor(colors[mod])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["flair", "t1ce"])
        ax.set_title(metric_name, fontsize=9)
        ax.set_ylabel(col, fontsize=8)
    fig.suptitle("Analysis 6: Modality Comparison (flair vs t1ce) — v2", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis6_modality_boxplot.png"), dpi=150)
    plt.close()

    print("  모달리티 비교 요약:")
    print(df_sum.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# 분석 7 (신규) — 슬라이스 크기 편향 검증
# ══════════════════════════════════════════════════════════════
def analysis7_pixel_bias(cases: List[dict]):
    """
    목적:
      slice_sample_counts_main 이 픽셀 수 누적이므로
      큰 슬라이스(픽셀 많은)가 자동으로 높은 카운트를 갖는다.
      이것이 Spearman_main_raw 를 얼마나 왜곡하는지 직접 측정.

    측정 지표:
      1. Spearman(pixel_counts, main_counts_raw)
         → 픽셀 수와 카운트의 상관: 1에 가까울수록 카운트가 픽셀 크기에 지배됨
      2. Spearman(pixel_counts, residuals)
         → 픽셀 수와 잔차의 상관: 경계 슬라이스(작음, 잔차 높음) 가설 검증
      3. Spearman(pixel_counts, main_counts_corrected)
         → 보정 후 남은 픽셀 편향 (0에 가까워야 보정이 완전함)
    """
    print("=== 분석 7: 슬라이스 크기 편향 검증 ===")
    rows = []
    for c in cases:
        r   = c["slice_residuals_final"].astype(np.float64)
        cm  = c["slice_sample_counts_main"].astype(np.float64)
        pix = c["slice_pixel_counts"].astype(np.float64)
        cm_corr = correct_counts_by_pixel(cm, pix)

        rho_pix_cm_raw,   _ = stats.spearmanr(pix, cm)
        rho_pix_resid,    _ = stats.spearmanr(pix, r)
        rho_pix_cm_corr,  _ = stats.spearmanr(pix, cm_corr)

        rows.append({
            "patient":            c["patient"],
            "modality":           c["modality"],
            "rho_pix_cm_raw":     rho_pix_cm_raw,
            "rho_pix_resid":      rho_pix_resid,
            "rho_pix_cm_corr":    rho_pix_cm_corr,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis7_pixel_bias.csv"), index=False)

    # 그림 7-A: 세 Spearman 분포 비교 (flair / t1ce 각 subplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_7 = {
        "ρ(pixel, cm_raw)":   ("rho_pix_cm_raw",  "#e74c3c"),
        "ρ(pixel, residual)": ("rho_pix_resid",   "#3498db"),
        "ρ(pixel, cm_corr)":  ("rho_pix_cm_corr", "#2ecc71"),
    }
    for ax, mod in zip(axes, MODALITIES):
        sub   = df[df["modality"] == mod]
        data  = [sub[col].values for _, (col, _) in metrics_7.items()]
        labels = list(metrics_7.keys())
        colors = [c for _, (_, c) in metrics_7.items()]
        bp     = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Spearman correlation")
        ax.set_title(f"{mod.upper()}")
        means = [sub[col].mean() for _, (col, _) in metrics_7.items()]
        for i, m in enumerate(means, start=1):
            ax.text(i, m + 0.03, f"{m:.3f}", ha="center", fontsize=8)
    fig.suptitle("Analysis 7: Slice Pixel Count Bias Decomposition\n"
                 "ρ(pixel, cm_raw) ≈ 1 → raw counts dominated by slice size\n"
                 "ρ(pixel, cm_corr) ≈ 0 → correction successful")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis7a_pixel_bias_boxplot.png"), dpi=150)
    plt.close()

    # 그림 7-B: 대표 케이스 — 픽셀 수 vs 잔차 scatter (경계 슬라이스 가설)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df   = df[df["modality"] == mod]
        rep_row  = sub_df.iloc[sub_df["rho_pix_resid"].abs().argsort().iloc[-1]]
        rep_case = next(c for c in cases
                        if c["patient"] == rep_row["patient"]
                        and c["modality"] == mod)
        pix = rep_case["slice_pixel_counts"].astype(np.float64)
        r   = rep_case["slice_residuals_final"].astype(np.float64)
        ax.scatter(pix, r, alpha=0.5, s=20,
                   color="steelblue" if mod == "flair" else "tomato")
        ax.set_xlabel("Slice pixel count")
        ax.set_ylabel("Slice residual (EMA)")
        ax.set_title(f"{mod.upper()} — Patient {rep_row['patient']}\n"
                     f"ρ(pixel, residual)={rep_row['rho_pix_resid']:.3f}")
    fig.suptitle("Analysis 7B: Pixel Count vs Residual\n"
                 "(음수 상관 → 경계 슬라이스(작음)가 높은 잔차를 가짐)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis7b_pixel_residual_scatter.png"), dpi=150)
    plt.close()

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
    analysis4_alpha_simulation(cases)
    df5 = analysis5_hard_easy_slices(cases)
    analysis6_modality_comparison(df1, df2, df3, df5)
    analysis7_pixel_bias(cases)

    print(f"\n✅ 모든 분석 완료. 결과 저장 위치: {OUT_DIR}")

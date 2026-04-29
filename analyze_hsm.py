#!/usr/bin/env python3
"""
analyze_hsm.py
Hard Slice Mining 분석 스크립트 — 분석 1~6 전체
사용법: python analyze_hsm.py
출력:  ./hsm_analysis_output/ 아래 CSV + PNG
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.special import rel_entr   # KL-divergence

# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
RECON_ROOT  = "/dshome/ddualab/dongnyeok/NeSVoR/recon_analysis"
TEST_CSV    = "/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
OUT_DIR     = "./hsm_analysis_output"
MODALITIES  = ["flair", "t1ce"]
EMA_ALPHA   = 0.99       # 실제 사용된 값
SIM_ALPHAS  = [0.80, 0.90, 0.95, 0.99]   # 분석 4 시뮬레이션 대상
N_ITER      = 2000

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════
def load_case(patient_dir: str, patient_id: str, modality: str) -> Optional[dict]:
    """케이스 하나의 npy 파일을 모두 읽어 dict로 반환. 파일 누락 시 None."""
    stem = f"{patient_id}_{modality}_4x5_analysis"
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


def load_all_cases() -> list[dict]:
    df_ids = pd.read_csv(TEST_CSV)
    full_ids = df_ids.iloc[:, 0].str.strip().tolist()
    cases = []
    for full_id in full_ids:
        patient_id = full_id.split("_")[-1]
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

    # 그림 1-A: CV 분포 (flair vs t1ce)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["cv"]
        ax.hist(sub, bins=20, edgecolor="white", color="steelblue" if mod == "flair" else "tomato")
        ax.axvline(sub.mean(), color="black", linestyle="--", label=f"mean={sub.mean():.3f}")
        ax.set_title(f"CV Distribution — {mod.upper()}")
        ax.set_xlabel("Coefficient of Variation (std/mean)")
        ax.set_ylabel("# cases")
        ax.legend()
    fig.suptitle("Analysis 1: Residual CV per Case\n(낮을수록 hard/easy 구분 불가 → HSM ≈ uniform)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis1_cv_histogram.png"), dpi=150)
    plt.close()

    # 그림 1-B: 대표 케이스 잔차 분포 (각 모달리티 CV 중앙값 케이스)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df = df[df["modality"] == mod]
        median_cv = sub_df["cv"].median()
        rep_row = sub_df.iloc[(sub_df["cv"] - median_cv).abs().argsort().iloc[0]]
        rep_case = next(c for c in cases
                        if c["patient"] == rep_row["patient"] and c["modality"] == mod)
        r = rep_case["slice_residuals_final"]
        ax.hist(r, bins=30, edgecolor="white", color="steelblue" if mod == "flair" else "tomato")
        ax.set_title(f"{mod.upper()} — Patient {rep_row['patient']}\nCV={rep_row['cv']:.3f} (median)")
        ax.set_xlabel("Slice Residual (EMA)")
        ax.set_ylabel("# slices")
    fig.suptitle("Analysis 1: Representative Residual Histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis1_residual_histogram_repr.png"), dpi=150)
    plt.close()

    print(f"  CV 전체 평균: flair={df[df.modality=='flair']['cv'].mean():.4f}, "
          f"t1ce={df[df.modality=='t1ce']['cv'].mean():.4f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 2 — 샘플링 편향 (잔차 vs 실제 카운트)
# ══════════════════════════════════════════════════════════════
def analysis2_sampling_bias(cases: List[dict]):
    print("=== 분석 2: 샘플링 편향 ===")
    rows = []
    for c in cases:
        r   = c["slice_residuals_final"].astype(np.float64)
        cm  = c["slice_sample_counts_main"].astype(np.float64)
        cp  = c["slice_sample_counts_patch"].astype(np.float64)

        # 이론적 샘플링 확률 (잔차 기반)
        r_prob = r / (r.sum() + 1e-12)
        # 실제 샘플링 비율
        cm_prob = cm / (cm.sum() + 1e-12)
        cp_prob = cp / (cp.sum() + 1e-12)

        # Spearman 상관
        rho_main,  _ = stats.spearmanr(r, cm)
        rho_patch, _ = stats.spearmanr(r, cp)

        # KL-divergence: KL(r_prob || cm_prob)  (uniform이면 0에 가까워야 정상)
        # uniform 기준값 계산
        n = len(r)
        uniform = np.ones(n) / n
        eps = 1e-12
        kl_main  = np.sum(rel_entr(r_prob + eps, cm_prob  + eps))
        kl_patch = np.sum(rel_entr(r_prob + eps, cp_prob  + eps))
        kl_unif  = np.sum(rel_entr(r_prob + eps, uniform  + eps))  # 이론적 최대 편향

        rows.append({
            "patient":   c["patient"],
            "modality":  c["modality"],
            "spearman_main":   rho_main,
            "spearman_patch":  rho_patch,
            "kl_main":         kl_main,
            "kl_patch":        kl_patch,
            "kl_uniform_ref":  kl_unif,   # 비교 기준 (완전 균등 샘플링 시)
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis2_sampling_bias.csv"), index=False)

    # 그림 2-A: Spearman 상관 분포
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]
        ax.scatter(sub["spearman_main"], sub["spearman_patch"], alpha=0.6, s=30)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("Spearman(residual, main_counts)")
        ax.set_ylabel("Spearman(residual, patch_counts)")
        ax.set_title(f"{mod.upper()}\n(1.0 = 잔차대로 샘플링, 0 = 무관)")
    fig.suptitle("Analysis 2: Residual–SamplingCount Correlation\n(값이 낮으면 HSM이 샘플링을 실질적으로 안 바꿨음)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis2_spearman_scatter.png"), dpi=150)
    plt.close()

    print(f"  Spearman(main) 평균: flair={df[df.modality=='flair']['spearman_main'].mean():.3f}, "
          f"t1ce={df[df.modality=='t1ce']['spearman_main'].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════
# 분석 3 — EMA 수렴 속도 & alpha=0.99 문제 진단
# ══════════════════════════════════════════════════════════════
def analysis3_ema_convergence(cases: List[dict]):
    print("=== 분석 3: EMA 수렴 속도 ===")
    rows = []
    for c in cases:
        hist_iters  = c["residuals_history_iters"]    # shape (T,)
        hist_values = c["residuals_history_values"]   # shape (T, n_slices)
        n_slices = hist_values.shape[1]

        # 스냅샷별 잔차 분산 (시계열)
        var_series  = hist_values.var(axis=1)         # (T,)
        # 정규화: 각 케이스 최대 분산 기준
        var_norm = var_series / (var_series.max() + 1e-12)

        # 초기 → 최종 분산 변화율
        delta_var = var_series[-1] - var_series[0]

        # 슬라이스당 평균 업데이트 횟수 추정
        cm = c["slice_sample_counts_main"].astype(np.float64)
        # main 배치에서 슬라이스 1개가 업데이트된 총 횟수
        # = 총 픽셀 샘플 / 슬라이스당 평균 픽셀 → 근사치
        pix = c["slice_pixel_counts"].astype(np.float64)
        avg_pix_per_slice = pix.mean()
        total_main_pixels = cm.sum()   # 2000 iter × batch_size(4096)에 가까울 것
        avg_update_per_slice = (total_main_pixels / avg_pix_per_slice) / n_slices

        # alpha=0.99 기준 초기값 잔존율
        # r_k ≈ alpha^k * r_0 + (1-alpha^k) * r_true
        init_retention = EMA_ALPHA ** avg_update_per_slice

        rows.append({
            "patient":              c["patient"],
            "modality":             c["modality"],
            "n_slices":             n_slices,
            "avg_update_per_slice": avg_update_per_slice,
            "init_retention_0.99": init_retention,
            "var_initial":          var_series[0],
            "var_final":            var_series[-1],
            "delta_var":            delta_var,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis3_ema_convergence.csv"), index=False)

    # 그림 3-A: 슬라이스당 평균 업데이트 횟수 분포
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]
        ax.hist(sub["avg_update_per_slice"], bins=20, edgecolor="white",
                color="steelblue" if mod == "flair" else "tomato")
        ax.axvline(sub["avg_update_per_slice"].mean(), color="black", linestyle="--",
                   label=f"mean={sub['avg_update_per_slice'].mean():.1f}")
        ax.set_xlabel("Avg # EMA updates per slice")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()} — EMA Update Count per Slice")
        ax.legend()
    fig.suptitle("Analysis 3A: How many times was each slice updated in EMA?\n"
                 "(적을수록 alpha=0.99 EMA가 초기값에 고착됨)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis3a_update_count.png"), dpi=150)
    plt.close()

    # 그림 3-B: 초기값 잔존율 분포 (alpha=0.99)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub = df[df["modality"] == mod]["init_retention_0.99"]
        ax.hist(sub, bins=20, edgecolor="white",
                color="steelblue" if mod == "flair" else "tomato")
        ax.axvline(sub.mean(), color="black", linestyle="--", label=f"mean={sub.mean():.3f}")
        ax.axvline(0.5, color="red", linestyle=":", label="50% threshold")
        ax.set_xlabel("Initial value retention  alpha^k")
        ax.set_ylabel("# cases")
        ax.set_title(f"{mod.upper()} — Initial Retention (alpha=0.99)")
        ax.legend()
    fig.suptitle("Analysis 3B: alpha=0.99 → 초기값이 얼마나 남아있나?\n"
                 "(0.5 이상 = EMA가 실제 잔차보다 초기값에 더 가까움)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis3b_init_retention.png"), dpi=150)
    plt.close()

    # 그림 3-C: 대표 케이스 잔차 분산 시계열
    # flair, t1ce 각 1개씩 — n_slices 중앙값 케이스
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, MODALITIES):
        sub_df = df[df["modality"] == mod]
        med_n  = sub_df["n_slices"].median()
        rep_row = sub_df.iloc[(sub_df["n_slices"] - med_n).abs().argsort().iloc[0]]
        rep_case = next(c for c in cases
                        if c["patient"] == rep_row["patient"] and c["modality"] == mod)
        hist_iters  = rep_case["residuals_history_iters"]
        hist_values = rep_case["residuals_history_values"]
        var_series  = hist_values.var(axis=1)
        ax.plot(hist_iters, var_series, marker="o", color="steelblue" if mod == "flair" else "tomato")
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
    return df


# ══════════════════════════════════════════════════════════════
# 분석 4 — Alpha 하이퍼파라미터 시뮬레이션
# ══════════════════════════════════════════════════════════════
def analysis4_alpha_simulation(cases: List[dict]):
    """
    수집된 잔차 시계열 스냅샷을 이용해, 다른 alpha 값으로 EMA를 재계산했을 때
    잔차 분산이 어떻게 달라졌을지 시뮬레이션.

    핵심 아이디어:
      스냅샷 t에서 t+1 사이 실제 잔차 변화량을 역산해서
      'raw per-snapshot signal'을 복원한 뒤, 다른 alpha로 재적분.
    """
    print("=== 분석 4: Alpha 시뮬레이션 ===")

    # 전체 케이스 평균 분산 시계열을 alpha별로 계산
    # shape: {alpha: (T,)}  — T = 스냅샷 수
    T_max = min(c["residuals_history_iters"].shape[0] for c in cases)

    alpha_var_curves = {a: [] for a in SIM_ALPHAS}

    for c in cases:
        hist_values = c["residuals_history_values"][:T_max]   # (T, n_slices)
        hist_iters  = c["residuals_history_iters"][:T_max]
        T, n_slices = hist_values.shape

        # 스냅샷 간 간격 (iter 수)
        delta_iters = np.diff(hist_iters, prepend=0)   # 첫 스냅샷까지의 간격 포함

        # alpha=0.99 기준 스냅샷에서 'effective new signal' 역산
        # snapshot[t] ≈ alpha^(delta) * snapshot[t-1] + (1 - alpha^(delta)) * signal[t]
        # → signal[t] = (snapshot[t] - alpha^(delta) * snapshot[t-1]) / (1 - alpha^(delta))
        signals = np.zeros_like(hist_values)
        signals[0] = hist_values[0]  # 초기 snapshot = 초기 signal
        for t in range(1, T):
            d    = delta_iters[t]
            aD   = EMA_ALPHA ** d
            denom = 1.0 - aD
            if denom < 1e-10:
                signals[t] = hist_values[t]
            else:
                signals[t] = (hist_values[t] - aD * hist_values[t-1]) / denom
            # 음수 클리핑 (역산 노이즈 방지)
            signals[t] = np.clip(signals[t], 0, None)

        # 각 alpha로 EMA 재계산
        for alpha in SIM_ALPHAS:
            ema = signals[0].copy()
            var_curve = []
            for t in range(T):
                d  = delta_iters[t]
                aD = alpha ** d
                ema = aD * ema + (1.0 - aD) * signals[t]
                var_curve.append(ema.var())
            alpha_var_curves[alpha].append(var_curve)

    # 케이스 평균
    mean_curves = {}
    for alpha in SIM_ALPHAS:
        arr = np.array(alpha_var_curves[alpha])   # (n_cases, T)
        mean_curves[alpha] = arr.mean(axis=0)

    # iter 축은 첫 번째 케이스 기준
    ref_iters = cases[0]["residuals_history_iters"][:T_max]

    # 그림 4: alpha별 평균 분산 시계열
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    for alpha, color in zip(SIM_ALPHAS, colors):
        lw = 2.5 if alpha == EMA_ALPHA else 1.5
        ls = "--" if alpha == EMA_ALPHA else "-"
        label = f"alpha={alpha}" + (" (현재)" if alpha == EMA_ALPHA else "")
        ax.plot(ref_iters, mean_curves[alpha], label=label, color=color, lw=lw, ls=ls)

    ax.set_xlabel("Training iter")
    ax.set_ylabel("Avg variance of slice residuals (across 96 cases)")
    ax.set_title("Analysis 4: Alpha Simulation — Residual Variance over Training\n"
                 "(분산 높을수록 HSM이 슬라이스를 잘 차별화함)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis4_alpha_simulation.png"), dpi=150)
    plt.close()

    # CSV 저장
    df_out = pd.DataFrame({"iter": ref_iters})
    for alpha in SIM_ALPHAS:
        df_out[f"var_alpha_{alpha}"] = mean_curves[alpha]
    df_out.to_csv(os.path.join(OUT_DIR, "analysis4_alpha_simulation.csv"), index=False)

    # 이론적 초기값 잔존율 비교표
    update_counts = [5, 10, 15, 20, 30, 50]
    rows = []
    for k in update_counts:
        row = {"avg_updates_per_slice": k}
        for alpha in SIM_ALPHAS:
            row[f"retention_alpha_{alpha}"] = round(alpha ** k, 4)
        rows.append(row)
    df_ret = pd.DataFrame(rows)
    df_ret.to_csv(os.path.join(OUT_DIR, "analysis4_retention_table.csv"), index=False)

    # 그림 4-B: 이론적 잔존율 비교 (라인 플롯)
    fig, ax = plt.subplots(figsize=(10, 5))
    k_arr = np.arange(1, 51)
    for alpha, color in zip(SIM_ALPHAS, colors):
        lw = 2.5 if alpha == EMA_ALPHA else 1.5
        ls = "--" if alpha == EMA_ALPHA else "-"
        ax.plot(k_arr, alpha ** k_arr,
                label=f"alpha={alpha}" + (" (현재)" if alpha == EMA_ALPHA else ""),
                color=color, lw=lw, ls=ls)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", label="50% retention")
    ax.set_xlabel("# EMA updates per slice (k)")
    ax.set_ylabel("Initial value retention  alpha^k")
    ax.set_title("Analysis 4B: Theoretical Initial Retention by Alpha\n"
                 "(슬라이스당 업데이트 횟수 = N_iter / n_slices 추정)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis4b_retention_curve.png"), dpi=150)
    plt.close()

    print(f"  alpha 시뮬레이션 완료. 최종 분산 비교:")
    for alpha in SIM_ALPHAS:
        print(f"    alpha={alpha}: final_var_mean={mean_curves[alpha][-1]:.6f}")


# ══════════════════════════════════════════════════════════════
# 분석 5 — Hard / Easy 슬라이스 특성
# ══════════════════════════════════════════════════════════════
def analysis5_hard_easy_slices(cases: List[dict]):
    print("=== 분석 5: Hard/Easy 슬라이스 특성 ===")
    rows = []
    for c in cases:
        r   = c["slice_residuals_final"]
        pix = c["slice_pixel_counts"].astype(np.float64)
        n   = len(r)
        top20_idx = np.argsort(r)[-max(1, n//5):]   # 상위 20% (hard)
        bot20_idx = np.argsort(r)[:max(1, n//5)]    # 하위 20% (easy)

        rows.append({
            "patient":              c["patient"],
            "modality":             c["modality"],
            "hard_mean_residual":   r[top20_idx].mean(),
            "easy_mean_residual":   r[bot20_idx].mean(),
            "hard_easy_ratio":      r[top20_idx].mean() / (r[bot20_idx].mean() + 1e-12),
            "hard_mean_pixels":     pix[top20_idx].mean(),
            "easy_mean_pixels":     pix[bot20_idx].mean(),
            "pixel_ratio_hard_easy": pix[top20_idx].mean() / (pix[bot20_idx].mean() + 1e-12),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis5_hard_easy.csv"), index=False)

    # 그림 5: hard/easy 잔차 비율 및 픽셀 수 비율
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(
        axes,
        ["hard_easy_ratio", "pixel_ratio_hard_easy"],
        ["Hard/Easy Residual Ratio", "Hard/Easy Pixel Count Ratio"],
    ):
        for mod, color in zip(MODALITIES, ["steelblue", "tomato"]):
            sub = df[df["modality"] == mod][col]
            ax.hist(sub, bins=20, alpha=0.6, label=mod, edgecolor="white", color=color)
        ax.axvline(1.0, color="black", ls="--", lw=1, label="ratio=1 (no diff)")
        ax.set_xlabel(col)
        ax.set_ylabel("# cases")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Analysis 5: Hard vs Easy Slice Characteristics\n"
                 "(잔차 비율 ≈ 1 → hard/easy 차이 미미)")
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
        "CV (잔차 변동계수)":         (df1,  "cv"),
        "Spearman (잔차-샘플링)":     (df2,  "spearman_main"),
        "Init Retention (alpha=0.99)": (df3,  "init_retention_0.99"),
        "Hard/Easy 잔차 비율":        (df5,  "hard_easy_ratio"),
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

    # 그림 6: 4개 지표 박스플롯 (flair vs t1ce 나란히)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = {"flair": "steelblue", "t1ce": "tomato"}
    for ax, (metric_name, (df, col)) in zip(axes.flat, metrics.items()):
        data_by_mod = [df[df["modality"] == mod][col].values for mod in MODALITIES]
        bp = ax.boxplot(data_by_mod, patch_artist=True, widths=0.5)
        for patch, mod in zip(bp["boxes"], MODALITIES):
            patch.set_facecolor(colors[mod])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["flair", "t1ce"])
        ax.set_title(metric_name)
        ax.set_ylabel(col)
    fig.suptitle("Analysis 6: Modality Comparison (flair vs t1ce)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis6_modality_boxplot.png"), dpi=150)
    plt.close()

    print("  모달리티 비교 요약:")
    print(df_sum.to_string(index=False))


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

    print(f"\n✅ 모든 분석 완료. 결과 저장 위치: {OUT_DIR}")

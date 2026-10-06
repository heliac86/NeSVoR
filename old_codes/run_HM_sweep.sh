#!/usr/bin/env bash
# ================================================================
# run_HM_sweep.sh  (v2 — 2026-04-08 수정)
# Hard Mining (HM2) + FF Loss Ablation 스윗
#
# ── 수정 이력 ──────────────────────────────────────────────────
# v1 오류: BASE_ARGS에 --patch-size, --n-patches, --weight-ff-loss,
#          --ff-alpha, --weight-diversity-loss, --target-diversity-var
#          가 누락되어 G5_lowvar 재현 불가 상태였음.
#          --gating-grad-clip도 0.0이어야 하는데 1.0으로 잘못 설정.
# v2 수정: G5_lowvar 완전 인자를 BASE_ARGS에 통합.
#          세트 재설계 (FF_w010 제거, no_FF ablation 추가).
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G7B 스윗 결과:
#   G5_lowvar : PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414  ← 최고
#   공간 적응형 게이팅(G6/G7), entropy/gini diversity loss 계열
#   모두 G5_lowvar를 넘지 못함 → 전역 gating + variance loss 유지.
#
# FF Loss는 G5_lowvar의 BASE 구성에 이미 포함되어 있음.
# 따라서 "FF 추가" 실험이 아니라 "FF 제거" ablation이 의미 있음.
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q0. 코드 변경(parsers.py 등) 후에도 G5_lowvar 수치가 유지되는가?
#     → G5_repro vs 기존 25.664
# Q1. HM2(hard-mining-main-loss) 단독이 G5_lowvar를 이기는가?
#     → HM2 vs G5_repro  (FF Loss는 켜진 채로 비교)
# Q2. FF Loss를 끄면 얼마나 나빠지는가? (역방향 ablation)
#     → no_FF vs G5_repro
# Q3. FF 없을 때 HM2가 FF 손실을 만회하는가?
#     → HM2_noFF vs no_FF, G5_repro
# Q4. warmup(500 iter) 단독이 G5_lowvar를 이기는가?
#     → warm500 vs G5_repro
#     ※ G7_h32_warm 이상치(24.126)는 z_repr 불안정이 원인.
#        전역 gating에서 warmup 단독 효과 재검증 필요.
# Q5. HM2 + warmup 조합이 각 단독보다 나은가?
#     → HM2_warm vs HM2, warm500
#
# ── 실험 세트 ──────────────────────────────────────────────────
#   [0] G5_repro   : G5_lowvar 완전 재현 (코드 드리프트 확인)
#   [1] HM2        : BASE + hard-mining-main-loss
#   [2] no_FF      : BASE - FF Loss (ablation)
#   [3] HM2_noFF   : BASE + HM2 - FF Loss
#   [4] warm500    : BASE + gating-warmup-iters=500
#   [5] HM2_warm   : BASE + HM2 + warmup=500
#
# 비교 기준선 (재실행 없이 기존 결과 재사용):
#   G5_lowvar : PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 6세트 x 15 = 90
#
# 사용법:
#   chmod +x run_HM_sweep.sh
#   ./run_HM_sweep.sh                        # 모든 세트 실행
#   ENABLE_G5_REPRO=0  ./run_HM_sweep.sh     # G5_repro 건너뜀
#   ENABLE_HM2=0       ./run_HM_sweep.sh     # HM2 건너뜀
#   ENABLE_NO_FF=0     ./run_HM_sweep.sh     # no_FF 건너뜀
#   ENABLE_HM2_NOFF=0  ./run_HM_sweep.sh     # HM2_noFF 건너뜀
#   ENABLE_WARM500=0   ./run_HM_sweep.sh     # warm500 건너뜀
#   ENABLE_HM2_WARM=0  ./run_HM_sweep.sh     # HM2_warm 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_G5_REPRO=${ENABLE_G5_REPRO:-1}
ENABLE_HM2=${ENABLE_HM2:-1}
ENABLE_NO_FF=${ENABLE_NO_FF:-1}
ENABLE_HM2_NOFF=${ENABLE_HM2_NOFF:-1}
ENABLE_WARM500=${ENABLE_WARM500:-1}
ENABLE_HM2_WARM=${ENABLE_HM2_WARM:-1}

# ── BASE_ARGS: G5_lowvar 완전 재현 인자 ───────────────────────
# run_G5_sweep.sh의 BASE_ARGS + G5_lowvar 세트별 인자를 통합.
# 이 스크립트의 모든 세트는 아래를 기준으로 ±변경만 적용.
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-ff-loss 0.10
  --ff-alpha 1.0
  --weight-image 1.0
  --delta 0.05
  --weight-diversity-loss 0.05
  --target-diversity-var 0.12
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 0.0
"

# no_FF 세트용: FF Loss 관련 인자를 BASE에서 제거한 버전
# (--patch-size, --n-patches는 FF 없이도 무해하므로 유지)
BASE_NOFF_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --weight-image 1.0
  --delta 0.05
  --weight-diversity-loss 0.05
  --target-diversity-var 0.12
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 0.0
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ──────────────────────────────────────
run_case() {
    local EXP_TAG=$1
    local PATIENT=$2
    local MODALITY=$3
    shift 3
    local EXTRA_ARGS="$*"

    local INPUT="${DEGRADED_ROOT}/BraTS20_Training_${PATIENT}/BraTS20_Training_${PATIENT}_${MODALITY}.nii"
    local OUTPUT_VOL="${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local OUTPUT_MODEL="${PATIENT}_${MODALITY}_${EXP_TAG}_model.pt"

    echo ""
    echo "================================================================"
    echo "  [START] Tag=${EXP_TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --output-model  "${OUTPUT_MODEL}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        echo "  Vol   → ${OUTPUT_VOL}"
        echo "  Model → ${OUTPUT_MODEL}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# 세트 0: G5_repro
#
#   목적: parsers.py 등 코드 변경 이후에도 G5_lowvar 수치
#         (PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414)가
#         재현되는지 확인.
#         드리프트 발생 시 이후 결과 전체를 신뢰할 수 없음.
#   변경점: 없음 (BASE_ARGS 그대로)
# ================================================================
if [ "${ENABLE_G5_REPRO}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 0: G5_repro  (코드 변경 후 재현성 확인)"
    echo "  G5_lowvar 완전 재현 — BASE_ARGS 그대로"
    echo "  기준: PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G5_repro" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS}
        done
    done
else
    echo "  [SKIP] G5_repro (ENABLE_G5_REPRO=0)"
fi

# ================================================================
# 세트 1: HM2
#
#   목적: --hard-mining-main-loss 단독 효과 측정.
#         FF Loss는 BASE에 포함된 채로 유지하고,
#         get_batch()에 slice_residuals EMA 기반 hard mining만 추가.
#         어려운 슬라이스를 더 자주 샘플링함으로써 메인 MSE
#         학습 수렴이 개선되는지 확인.
#   변경점: + --hard-mining-main-loss
#   비교:   G5_repro (HM2 없음) vs HM2 (HM2 있음)
# ================================================================
if [ "${ENABLE_HM2}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: HM2  (Hard Slice Mining for get_batch)"
    echo "  BASE_ARGS + --hard-mining-main-loss"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --hard-mining-main-loss
        done
    done
else
    echo "  [SKIP] HM2 (ENABLE_HM2=0)"
fi

# ================================================================
# 세트 2: no_FF
#
#   목적: FF Loss 제거 시 성능 하락 폭 측정 (역방향 ablation).
#         FF Loss가 G5_lowvar에서 얼마나 기여하는지 정량화.
#         만약 하락이 미미하다면 이후 실험에서 FF를 생략해
#         학습 속도를 높이는 선택지가 생김.
#   변경점: BASE에서 --weight-ff-loss, --ff-alpha 제거
#           (--patch-size, --n-patches도 제거 — FF 없으면 불필요)
#   비교:   G5_repro (FF 있음) vs no_FF (FF 없음)
# ================================================================
if [ "${ENABLE_NO_FF}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: no_FF  (FF Loss 제거 ablation)"
    echo "  BASE_NOFF_ARGS — weight-ff-loss, ff-alpha, patch 인자 제거"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "no_FF" "${PATIENT}" "${MODALITY}" \
                ${BASE_NOFF_ARGS}
        done
    done
else
    echo "  [SKIP] no_FF (ENABLE_NO_FF=0)"
fi

# ================================================================
# 세트 3: HM2_noFF
#
#   목적: FF Loss 없이 HM2만 있을 때의 성능 확인.
#         HM2가 FF Loss 손실을 만회할 수 있는지 파악.
#         no_FF 대비 HM2 추가 효과, G5_repro 대비 FF 제거+HM2 효과
#         두 가지를 동시에 읽을 수 있음.
#   변경점: BASE_NOFF_ARGS + --hard-mining-main-loss
#   비교:   no_FF vs HM2_noFF, G5_repro vs HM2_noFF
# ================================================================
if [ "${ENABLE_HM2_NOFF}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: HM2_noFF  (HM2 있음, FF Loss 없음)"
    echo "  BASE_NOFF_ARGS + --hard-mining-main-loss"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2_noFF" "${PATIENT}" "${MODALITY}" \
                ${BASE_NOFF_ARGS} \
                --hard-mining-main-loss
        done
    done
else
    echo "  [SKIP] HM2_noFF (ENABLE_HM2_NOFF=0)"
fi

# ================================================================
# 세트 4: warm500
#
#   목적: gating warmup 단독 효과 측정 (전역 gating 환경).
#         G7B의 G7_h32_warm 이상치(PSNR 24.126)는 warmup 중
#         불안정한 z_repr을 GatingMLP에 입력한 것이 원인.
#         전역 gating(level_weights)에서는 z_repr 없이
#         level_weights만 학습되므로 이상치 재발 가능성 낮음.
#         warmup의 순수 효과를 G5_lowvar 조건에서 재확인.
#   변경점: + --gating-warmup-iters 500
#   비교:   G5_repro (warmup 없음) vs warm500
# ================================================================
if [ "${ENABLE_WARM500}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 4: warm500  (density_net warmup freeze 500 iter)"
    echo "  BASE_ARGS + --gating-warmup-iters 500"
    echo "  ※ z_repr 없음 → G7_h32_warm 이상치 재발 가능성 낮음"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "warm500" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] warm500 (ENABLE_WARM500=0)"
fi

# ================================================================
# 세트 5: HM2_warm
#
#   목적: HM2 + warmup 조합 검증.
#         warmup 500 iter 동안 density_net이 안정화된 뒤
#         hard mining이 활성화되면 초기 불안정한 residuals로
#         인한 잘못된 샘플링 편향을 줄일 수 있음.
#         HM2 단독 대비 warmup을 선행하는 것이 실제로 도움이
#         되는지 확인.
#   변경점: + --hard-mining-main-loss + --gating-warmup-iters 500
#   비교:   HM2 vs HM2_warm, warm500 vs HM2_warm
# ================================================================
if [ "${ENABLE_HM2_WARM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 5: HM2_warm  (HM2 + warmup 500 조합)"
    echo "  BASE_ARGS + --hard-mining-main-loss + --gating-warmup-iters 500"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2_warm" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --hard-mining-main-loss \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] HM2_warm (ENABLE_HM2_WARM=0)"
fi

# ================================================================
# 평가 실행
# G5_lowvar 기존 결과는 파일 재사용 (재실행 없이 비교)
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags G5_lowvar \
               G5_repro \
               HM2 \
               no_FF \
               HM2_noFF \
               warm500 \
               HM2_warm \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_HM_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_HM_sweep.csv"
fi

# ── 최종 요약 ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  성공: ${SUCCESS} / ${TOTAL}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패한 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo "================================================================"

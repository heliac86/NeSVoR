#!/usr/bin/env bash
# ================================================================
# run_div_tighter_HM2.sh  (v1 — 2026-04-10)
# div_tighter + HM2 조합 단일 검증
#
# ── 실험 배경 ──────────────────────────────────────────────────
# div_sweep 결과:
#   div_tighter (weight=0.08, target=0.06) : PSNR 25.567 ← seed42 기준 최고
#   HM2_s42                                : PSNR 25.533
#   둘 다 t1ce에서 강한 신호 (+0.228, +0.170 dB)
#   → 조합 시 t1ce 추가 상승 or flair 손실 중첩 여부 확인 필요
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q. div_tighter + HM2 조합이 각 단독보다 나은가?
#    → div_tighter_HM2 vs div_tighter, HM2_s42, G5_lowvar_s42
#
# ── 비교에 사용할 기존 결과 (재실행 없음) ─────────────────────
#   G5_lowvar_s42 : PSNR 25.496 / SSIM 0.96029 / LPIPS 0.02430
#   HM2_s42       : PSNR 25.533 / SSIM 0.96048 / LPIPS 0.02406
#   div_tighter   : PSNR 25.567 / SSIM 0.96077 / LPIPS 0.02397
#   div_lowweight : PSNR 25.564 / SSIM 0.96049 / LPIPS 0.02417
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 1세트 x 15 = 15
#
# 사용법:
#   chmod +x run_div_tighter_HM2.sh
#   ./run_div_tighter_HM2.sh
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── BASE_ARGS: div_tighter 인자 (seed 42 포함) ────────────────
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
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 0.0
  --seed 42
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
# 세트 0: div_tighter_HM2
#
#   목적: div_tighter + hard-mining-main-loss 조합 검증.
#         둘 다 t1ce에서 강한 단독 신호를 보였으므로
#         조합 시 추가 상승 또는 flair 손실 중첩 여부 확인.
#   변경점: div_tighter BASE + --hard-mining-main-loss
# ================================================================
echo ""
echo "################################################################"
echo "  세트 0: div_tighter_HM2"
echo "  div_tighter BASE + --hard-mining-main-loss"
echo "  비교 기준 (기존 결과 재사용):"
echo "    G5_lowvar_s42 : PSNR 25.496"
echo "    HM2_s42       : PSNR 25.533"
echo "    div_tighter   : PSNR 25.567  ← 현재 seed42 최고"
echo "    div_lowweight : PSNR 25.564"
echo "################################################################"

for PATIENT in "${PATIENTS[@]}"; do
    for MODALITY in "${MODALITIES[@]}"; do
        run_case "div_tighter_HM2" "${PATIENT}" "${MODALITY}" \
            ${BASE_ARGS} \
            --hard-mining-main-loss
    done
done

# ================================================================
# 평가 실행
# 기존 결과(G5_lowvar_s42, HM2_s42, div_tighter, div_lowweight)는
# 파일 재사용 — 재실행 없이 비교
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags G5_lowvar_s42 \
               HM2_s42 \
               div_tighter \
               div_lowweight \
               div_tighter_HM2 \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_div_tighter_HM2.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_div_tighter_HM2.csv"
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

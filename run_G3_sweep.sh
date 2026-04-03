#!/usr/bin/env bash
# ================================================================
# run_G3_sweep.sh
# Gating G3 / warmup 실험 스윕
#
# 실험 세트:
#   G3_softvar          raw logit 기준 soft-target diversity loss
#   G3_softvar_softmax  softmax 출력 기준 soft-target diversity loss
#   G3_warmup           raw logit + density_net 500iter 동결
#
# 대상 환자: 003 026 030 040 060
# 모달리티 : flair / t2 / t1ce
# 총 케이스: 3세트 x 5명 x 3모달리티 = 45
#
# 사용법:
#   chmod +x run_G3_sweep.sh
#   ./run_G3_sweep.sh                    # 모든 세트 실행
#   ENABLE_SOFTVAR=0 ./run_G3_sweep.sh   # G3_softvar 건너뜀
#   ENABLE_SOFTMAX=0 ENABLE_WARMUP=0 ./run_G3_sweep.sh  # G3_softvar만 실행
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   예) 003_flair_4x5_G3_softvar.nii.gz
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 (0 = 건너뜀, 1 = 실행) ──────────────────────
# 환경변수로 덮어쓸 수 있음: ENABLE_SOFTVAR=0 ./run_G3_sweep.sh
ENABLE_SOFTVAR=${ENABLE_SOFTVAR:-1}    # G3_softvar
ENABLE_SOFTMAX=${ENABLE_SOFTMAX:-1}    # G3_softvar_softmax
ENABLE_WARMUP=${ENABLE_WARMUP:-1}      # G3_warmup

# ── 공통 고정 인자 ────────────────────────────────────────────────
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
  --weight-diversity-loss 0.01
  --gating-grad-clip 1.0
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ─────────────────────────────────────────
# 인자: $1=EXP_TAG  $2=PATIENT  $3=MODALITY  $4..=세트별 추가 인자
run_case() {
    local EXP_TAG=$1
    local PATIENT=$2
    local MODALITY=$3
    shift 3
    local EXTRA_ARGS="$*"

    local INPUT="${DEGRADED_ROOT}/BraTS20_Training_${PATIENT}/BraTS20_Training_${PATIENT}_${MODALITY}.nii"
    local OUTPUT="${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"

    echo ""
    echo "================================================================"
    echo "  [START] Tag=${EXP_TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${BASE_ARGS} \
        ${EXTRA_ARGS}

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# 세트 1: G3_softvar
#   raw logit 기준 soft-target diversity loss + grad clip
#   target_var=0.05 (G2_div001 에서 관찰된 안정 분산)
# ================================================================
if [ "${ENABLE_SOFTVAR}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: G3_softvar"
    echo "  --diversity-loss-space raw  --target-diversity-var 0.05"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G3_softvar" "${PATIENT}" "${MODALITY}" \
                --diversity-loss-space raw \
                --target-diversity-var 0.05
        done
    done
else
    echo ""
    echo "  [SKIP] G3_softvar (ENABLE_SOFTVAR=0)"
fi

# ================================================================
# 세트 2: G3_softvar_softmax
#   softmax 출력 기준 soft-target diversity loss
#   target_var=0.5 (softmax 공간에서의 적합한 타겟)
# ================================================================
if [ "${ENABLE_SOFTMAX}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: G3_softvar_softmax"
    echo "  --diversity-loss-space softmax  --target-diversity-var 0.5"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G3_softvar_softmax" "${PATIENT}" "${MODALITY}" \
                --diversity-loss-space softmax \
                --target-diversity-var 0.5
        done
    done
else
    echo ""
    echo "  [SKIP] G3_softvar_softmax (ENABLE_SOFTMAX=0)"
fi

# ================================================================
# 세트 3: G3_warmup
#   raw logit + density_net 500iter 동결 (warmup)
#   level_weights 가 먼저 분화한 뒤 density_net 이 적응하게 함
# ================================================================
if [ "${ENABLE_WARMUP}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: G3_warmup"
    echo "  --diversity-loss-space raw  --target-diversity-var 0.05  --gating-warmup-iters 500"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G3_warmup" "${PATIENT}" "${MODALITY}" \
                --diversity-loss-space raw \
                --target-diversity-var 0.05 \
                --gating-warmup-iters 500
        done
    done
else
    echo ""
    echo "  [SKIP] G3_warmup (ENABLE_WARMUP=0)"
fi

# ================================================================
# 평가 실행
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate5.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate5.py \
    --exp-tags H3 G1 G2_div001 G3_softvar G3_softvar_softmax G3_warmup \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_G3_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_G3_sweep.csv"
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

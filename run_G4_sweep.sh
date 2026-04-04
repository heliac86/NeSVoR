#!/usr/bin/env bash
# ================================================================
# run_G4_sweep.sh
# Gating G4 실험 스윗
#
# 대실험 배경:
#   G3 분석 결과, level_weights 가 자연스럽게 softmax_var ~ 0.01~0.04
#   수준에서 정체되어 있음. 목표:
#     1) weight를 5배로 놀려 더 강하게 분화를 유도 (G4_pushvar)
#     2) gating_grad_clip이 분화를 막고 있는지 확인 (G4_noclip)
#
# 실험 세트:
#   G4_pushvar   weight=0.05  target_var=0.20  grad_clip=1.0
#   G4_noclip    weight=0.05  target_var=0.20  grad_clip=0.0 (비활성화)
#
# 대상 환자: 003 026 030 040 060
# 모달리티 : flair / t2 / t1ce
# 총 케이스: 2세트 x 5명 x 3모달리티 = 30
#
# 특이사항:
#   --output-model 포함 → 학습 후 level_weights 직접 조회 가능
#
# 사용법:
#   chmod +x run_G4_sweep.sh
#   ./run_G4_sweep.sh                      # 모든 세트 실행
#   ENABLE_PUSHVAR=0 ./run_G4_sweep.sh     # G4_noclip 전용
#   ENABLE_NOCLIP=0  ./run_G4_sweep.sh     # G4_pushvar 전용
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
#   예) 003_flair_4x5_G4_pushvar.nii.gz
#       003_flair_G4_pushvar_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 (0 = 건너뜀, 1 = 실행) ─────────────────
ENABLE_PUSHVAR=${ENABLE_PUSHVAR:-1}    # G4_pushvar
ENABLE_NOCLIP=${ENABLE_NOCLIP:-1}      # G4_noclip

# ── 공통 고정 인자 ─────────────────────────────────────────
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
  --diversity-loss-space raw
  --target-diversity-var 0.20
  --weight-diversity-loss 0.05
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ──────────────────────────────────
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
        echo "  Model → ${OUTPUT_MODEL}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# 세트 1: G4_pushvar
#
#   동기: G2_div001의 sm_var 최대 상한이 ~0.038여서
#           분화가 비교적 약함. diversity loss를
#           weight=0.05(기존의 5배), target_var=0.20으로
#           휠씬 더 풍부한 분화를 유도.
#   grad_clip=1.0 유지 (안정성 보험)
# ================================================================
if [ "${ENABLE_PUSHVAR}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: G4_pushvar"
    echo "  --weight-diversity-loss 0.05"
    echo "  --target-diversity-var  0.20"
    echo "  --gating-grad-clip      1.0"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G4_pushvar" "${PATIENT}" "${MODALITY}" \
                --gating-grad-clip 1.0
        done
    done
else
    echo ""
    echo "  [SKIP] G4_pushvar (ENABLE_PUSHVAR=0)"
fi

# ================================================================
# 세트 2: G4_noclip
#
#   동기: gating_grad_clip=1.0이 level_weights의 gradient를
#           억제해 분화를 막고 있는지 확인.
#           clip을 완전히 제거하고 같은 weight/target로
#           얼마나 더 많이 분화되는지 측정.
#   grad_clip=0.0 (비활성화)
# ================================================================
if [ "${ENABLE_NOCLIP}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: G4_noclip"
    echo "  --weight-diversity-loss 0.05"
    echo "  --target-diversity-var  0.20"
    echo "  --gating-grad-clip      0.0  (비활성화)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G4_noclip" "${PATIENT}" "${MODALITY}" \
                --gating-grad-clip 0.0
        done
    done
else
    echo ""
    echo "  [SKIP] G4_noclip (ENABLE_NOCLIP=0)"
fi

# ================================================================
# 평가 실행
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags H3 G2_div001 G4_pushvar G4_noclip \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_G4_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_G4_sweep.csv"
fi

# ── 최종 요약 ────────────────────────────────────────────────
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

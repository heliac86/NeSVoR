#!/usr/bin/env bash
# ================================================================
# run_G5_sweep.sh
# Gating G5 실험 스윗
#
# 대실험 배경:
#   G4_noclip(w=0.05, tv=0.20, clip=0.0)이 G2_div001 대비
#   overall +0.056 dB 개선. 두 가지 미해결 질문 검증:
#
#   Q1. clip 제거 효과는 weight=0.01에서도 유효한가?
#       (G5_noclip_low: w=0.01, tv=0.20, clip=0.0)
#
#   Q2. target_var의 최적점은 0.20인가?
#       더 낙으면(G5_lowvar: tv=0.12) t1ce LPIPS 악화가 해소되는가?
#       더 높으면(G5_midvar: tv=0.30) 성능이 더 올라가는가?
#
# 실험 세트:
#   G5_noclip_low   w=0.01  tv=0.20  clip=0.0
#   G5_lowvar       w=0.05  tv=0.12  clip=0.0
#   G5_midvar       w=0.05  tv=0.30  clip=0.0
#
# 대상 환자: 003 026 030 040 060
# 모달리티 : flair / t2 / t1ce
# 총 케이스: 3세트 x 5명 x 3모달리티 = 45
#
# 사용법:
#   chmod +x run_G5_sweep.sh
#   ./run_G5_sweep.sh                        # 모든 세트 실행
#   ENABLE_NOCLIP_LOW=0 ./run_G5_sweep.sh   # G5_noclip_low 건너뜀
#   ENABLE_LOWVAR=0     ./run_G5_sweep.sh   # G5_lowvar 건너뜀
#   ENABLE_MIDVAR=0     ./run_G5_sweep.sh   # G5_midvar 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_NOCLIP_LOW=${ENABLE_NOCLIP_LOW:-1}
ENABLE_LOWVAR=${ENABLE_LOWVAR:-1}
ENABLE_MIDVAR=${ENABLE_MIDVAR:-1}

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
  --gating-grad-clip 0.0
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
# 세트 1: G5_noclip_low
#
#   검증 목적: clip 제거 효과가 weight=0.01에서도 유효한가?
#   G2_div001(w=0.01, clip=1.0) vs G5_noclip_low(w=0.01, clip=0.0)
#   나머지 조건이 같으므로 clip 효과만 분리 가능
# ================================================================
if [ "${ENABLE_NOCLIP_LOW}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: G5_noclip_low"
    echo "  --weight-diversity-loss 0.01"
    echo "  --target-diversity-var  0.20"
    echo "  --gating-grad-clip      0.0"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G5_noclip_low" "${PATIENT}" "${MODALITY}" \
                --weight-diversity-loss 0.01 \
                --target-diversity-var 0.20
        done
    done
else
    echo "  [SKIP] G5_noclip_low (ENABLE_NOCLIP_LOW=0)"
fi

# ================================================================
# 세트 2: G5_lowvar
#
#   검증 목적: target_var를 0.12로 낙춰서
#             G4_noclip에서 관찰된 t1ce LPIPS 안화가
#             해소되는지 확인
#   G4_noclip(tv=0.20) vs G5_lowvar(tv=0.12)
# ================================================================
if [ "${ENABLE_LOWVAR}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: G5_lowvar"
    echo "  --weight-diversity-loss 0.05"
    echo "  --target-diversity-var  0.12"
    echo "  --gating-grad-clip      0.0"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G5_lowvar" "${PATIENT}" "${MODALITY}" \
                --weight-diversity-loss 0.05 \
                --target-diversity-var 0.12
        done
    done
else
    echo "  [SKIP] G5_lowvar (ENABLE_LOWVAR=0)"
fi

# ================================================================
# 세트 3: G5_midvar
#
#   검증 목적: target_var를 0.30으로 높였을 때
#             성능이 더 올라가는지 확인
#             분화를 더 강하게 밀었을 때의 방향 파악
#   G4_noclip(tv=0.20) vs G5_midvar(tv=0.30)
# ================================================================
if [ "${ENABLE_MIDVAR}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: G5_midvar"
    echo "  --weight-diversity-loss 0.05"
    echo "  --target-diversity-var  0.30"
    echo "  --gating-grad-clip      0.0"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G5_midvar" "${PATIENT}" "${MODALITY}" \
                --weight-diversity-loss 0.05 \
                --target-diversity-var 0.30
        done
    done
else
    echo "  [SKIP] G5_midvar (ENABLE_MIDVAR=0)"
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
    --exp-tags H3 G2_div001 G4_noclip G5_noclip_low G5_lowvar G5_midvar \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_G5_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_G5_sweep.csv"
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

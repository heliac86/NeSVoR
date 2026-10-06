#!/usr/bin/env bash
# ================================================================
# run_H_sweep.sh
# weight-ff-loss 탐색 (H1=0.01 / H2=0.03 / H3=0.10)
# 대상 환자: 003 026 030 040 060
# 모달리티: flair / t2 / t1ce
# 쳙 5명 x 3모달리티 x 3조건 = 45 케이스
#
# 사용법:
#   chmod +x run_H_sweep.sh
#   ./run_H_sweep.sh
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   예) 003_flair_4x5_H1.nii.gz
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 공통 고정 인자 (모든 조건에서 동일) ──────────────────────────────
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --ff-alpha 1.0
  --weight-image 1.0
  --delta 0.05
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

# ── 탐색 조건 정의 ─────────────────────────────────────────────
# EXP_TAG 배열과 weight-ff-loss 배열을 인덱스로 매핑
declare -a EXP_TAGS=("H1"   "H2"   "H3"  )
declare -a FF_WEIGHTS=("0.01" "0.03" "0.10")

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ──────────────────────────────────────────
run_case() {
    local EXP_TAG=$1
    local FF_WEIGHT=$2
    local PATIENT=$3
    local MODALITY=$4

    local INPUT="${DEGRADED_ROOT}/BraTS20_Training_${PATIENT}/BraTS20_Training_${PATIENT}_${MODALITY}.nii"
    local OUTPUT="${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"

    echo ""
    echo "================================================================"
    echo "  [START] Tag=${EXP_TAG}  weight-ff-loss=${FF_WEIGHT}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        --weight-ff-loss ${FF_WEIGHT} \
        ${BASE_ARGS}

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

# ── 메인 루프: 조건 → 환자 → 모달리티 ──────────────────────────────
# 조건을 가장 바깥쪽 루프로 두어 조건 단위로 결과를 판단하기 쓰운 구조
for idx in 0 1 2; do
    TAG="${EXP_TAGS[$idx]}"
    FFW="${FF_WEIGHTS[$idx]}"

    echo ""
    echo "################################################################"
    echo "  조건: ${TAG}  (--weight-ff-loss ${FFW})"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "${TAG}" "${FFW}" "${PATIENT}" "${MODALITY}"
        done
    done
done

# ── 최종 요약 ──────────────────────────────────────────────────────
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

#!/usr/bin/env bash
# ================================================================
# run_G14s_abl.sh  (v1 — 2026-06-01)
# 세트: G14s_no_ff, G14s_no_gating, G14s_no_all
#
# ── 실험 목적 ─────────────────────────────────────────────────
# G14_silu (풀 모델) 기준 ablation study
#   G14s_no_ff      : FF Loss 제거 (--weight-ff-loss 0.0)
#   G14s_no_gating  : Gating 전체 제거 (--no-gating)
#   G14s_no_all     : FF Loss + Gating 모두 제거 (순수 베이스라인)
#
# ── G14_silu 풀 모델 ARGS (참고) ──────────────────────────────
#   COMMON_ARGS + --gating-activation silu
#   FF Loss    : --weight-ff-loss 0.05 --ff-alpha 2.0
#   Gating     : --spatial-gating --gating-hidden-dim 8
#                --gating-activation silu
#                --gating-lr-scale 2.0 --gating-grad-clip 0.1
#
# ── 사용 예 ──────────────────────────────────────────────────
#   ./run_G14s_abl.sh
#   SKIP_TRAIN=1 ./run_G14s_abl.sh   # 평가만
#   SKIP_EVAL=1  ./run_G14s_abl.sh   # 학습만
#
# ── 출력 파일 명명 규칙 ───────────────────────────────────────
#   recon_G14s_abl/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.pt
#
# ── 비교 기준점 (96-case) ────────────────────────────────────
#   G14_silu:  FLAIR 27.218 / T1CE 26.623  LPIPS F 0.0284 / T 0.0456
#   G12_h8z:   FLAIR 27.177 / T1CE 26.603  LPIPS F 0.0287 / T 0.0458
#   P2_LR2:    FLAIR 26.845 / T1CE 26.373  LPIPS F 0.0266 / T 0.0445
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s_abl"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# 비교용 기준 경로
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"
G12_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G12/G12_h8z"
G14S_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G14/G14_silu"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── G14_silu 공통 BASE_ARGS ───────────────────────────────────
COMMON_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-image 1.0
  --delta 0.05
  --ff-alpha 2.0
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --target-diversity-var 0.06
  --gating-grad-clip 0.1
  --gating-lr-scale 2.0
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
"
# ※ --weight-ff-loss, --spatial-gating, --gating-hidden-dim,
#    --gating-activation 은 세트별 EXTRA_ARGS에서 on/off 제어

# ── 세트별 추가 인자 정의 ────────────────────────────────────
declare -A EXTRA_ARGS

# FF Loss만 제거 / Gating 유지 (silu)
EXTRA_ARGS["G14s_no_ff"]="
  --weight-ff-loss 0.0
  --spatial-gating
  --gating-hidden-dim 8
  --gating-activation silu
"

# Gating만 제거 / FF Loss 유지
EXTRA_ARGS["G14s_no_gating"]="
  --weight-ff-loss 0.05
  --no-gating
"

# FF Loss + Gating 모두 제거 (순수 베이스라인)
EXTRA_ARGS["G14s_no_all"]="
  --weight-ff-loss 0.0
  --no-gating
"

# ── 실행 세트 ─────────────────────────────────────────────────
RUN_SETS=("G14s_no_ff" "G14s_no_gating" "G14s_no_all")

# ── 전체 환자 ID 로드 (96 cases) ─────────────────────────────
mapfile -t ALL_IDS < <(
    tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r' | grep -v '^$'
)

echo ""
echo "================================================================"
echo "  G14s Ablation 스크립트"
echo "  전체 케이스: ${#ALL_IDS[@]}명 × 2 modalities = $((${#ALL_IDS[@]} * 2)) cases"
echo "  실행 세트: ${RUN_SETS[*]}"
echo "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_EVAL=${SKIP_EVAL}"
echo "================================================================"

# ================================================================
# 단일 케이스 실행 함수
# ================================================================
run_case() {
    local EXP_TAG=$1
    local FULL_ID=$2
    local MODALITY=$3
    local EXTRA=$4

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${EXP_TAG}/${FULL_ID}"
    local OUTPUT_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] 이미 존재: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        return 0
    fi

    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}_input_missing")
        return 0
    fi

    echo ""
    echo "  [START] ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --output-model  "${DST_PT}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${COMMON_ARGS} \
        ${EXTRA}

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"
    fi
}

# ================================================================
# 학습: 각 세트 × 96 cases
# ================================================================
FAILED_CASES=()
TOTAL=0
SUCCESS=0

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    for EXP_TAG in "${RUN_SETS[@]}"; do
        EXTRA="${EXTRA_ARGS[${EXP_TAG}]}"

        echo ""
        echo "################################################################"
        echo "  학습 시작: ${EXP_TAG}"
        echo "  추가 인자: ${EXTRA}"
        echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "################################################################"

        for FULL_ID in "${ALL_IDS[@]}"; do
            for MODALITY in "${MODALITIES[@]}"; do
                run_case "${EXP_TAG}" "${FULL_ID}" "${MODALITY}" "${EXTRA}"
                TOTAL=$((TOTAL + 1))
            done
        done

        echo ""
        echo "  [완료] ${EXP_TAG}  $(date '+%Y-%m-%d %H:%M:%S')"
    done

    SUCCESS=$(( TOTAL - ${#FAILED_CASES[@]} ))
    echo ""
    echo "  학습 전체 완료: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  성공: ${SUCCESS} / ${TOTAL}"
else
    echo "  [SKIP] 학습 (SKIP_TRAIN=1)"
fi

# ================================================================
# 평가: G14_silu 포함 전체 세트 비교
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    echo ""
    echo "################################################################"
    echo "  평가 시작"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    RECON_DIR_ARGS=(
        "${P2_LR2_RECON_ROOT}:P2_LR2"
        "${G12_RECON_ROOT}:G12_h8z"
        "${G14S_RECON_ROOT}:G14_silu"
    )
    for EVAL_TAG in "${RUN_SETS[@]}"; do
        RECON_DIR_ARGS+=("${RECON_ROOT}/${EVAL_TAG}:${EVAL_TAG}")
    done

    EVAL_CSV="eval_G14s_abl_full96.csv"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs "${RECON_DIR_ARGS[@]}" \
        --degraded-root "${DEGRADED_ROOT}" \
        --test-csv      "${TEST_CSV}" \
        --modalities    flair t1ce \
        --output-csv    "${EVAL_CSV}"

    if [ $? -eq 0 ]; then
        echo "  ✅ 평가 완료 → ${EVAL_CSV}"
        echo ""
        echo "  ── 비교 기준점 (96-case) ──────────────────────────────"
        echo "  G14_silu:  FLAIR 27.218 / T1CE 26.623  LPIPS F 0.0284 / T 0.0456"
        echo "  G12_h8z:   FLAIR 27.177 / T1CE 26.603  LPIPS F 0.0287 / T 0.0458"
        echo "  P2_LR2:    FLAIR 26.845 / T1CE 26.373  LPIPS F 0.0266 / T 0.0445"
        echo "  ──────────────────────────────────────────────────────"
    else
        echo "  ❌ 평가 실패: ${EVAL_CSV}"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 최종 요약 ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  [G14s_abl] 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  학습 성공: ${SUCCESS:-0} / ${TOTAL:-0}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo ""
echo "  ── 실험 메모 ─────────────────────────────────────────────"
echo "  기준 모델    : G14_silu (풀 모델)"
echo "  G14s_no_ff   : FF Loss 제거 (--weight-ff-loss 0.0)"
echo "  G14s_no_gating: Gating 전체 제거 (--no-gating)"
echo "  G14s_no_all  : FF Loss + Gating 모두 제거 (순수 베이스라인)"
echo "  출력 경로    : ${RECON_ROOT}"
echo "================================================================"

#!/usr/bin/env bash
# ================================================================
# run_G14s_GA.sh  (v1 — 2026-06-02)
# 세트: G14_silu (게이팅 가중치 분석을 위한 재구성)
#
# ── 실험 목적 ─────────────────────────────────────────────────
# G14_silu 베이스로 전체 4개 모달리티 모달리티 볼륨 + 게이팅 가중치들이 포함된 .pt 생성
# gating_weights 기록을 위해 models.py를 수정했기 때문에 전체 모달리티에 대해서 다시 추론해야 함
#
# ── 사용 예 ──────────────────────────────────────────────────
#   ./run_G14s_t1t2.sh
#   SKIP_TRAIN=1 ./run_G14s_GA.sh   # 평가만
#   SKIP_EVAL=1  ./run_G14s_GA.sh   # 학습만
#
# ── 출력 파일 명명 규칙 ───────────────────────────────────────
#   recon_G14s/BraTS20_Training_{PATIENT}/
#     {PATIENT}_t1_4x5_G14s.nii.gz
#     {PATIENT}_t1_4x5_G14s.pt
#     {PATIENT}_t2_4x5_G14s.nii.gz
#     {PATIENT}_t2_4x5_G14s.pt
#     {PATIENT}_t1ce_4x5_G14s.nii.gz
#     {PATIENT}_t1ce_4x5_G14s.pt
#     {PATIENT}_flair_4x5_G14s.nii.gz
#     {PATIENT}_flair_4x5_G14s.pt
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
EXP_TAG="G14s"
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_${EXP_TAG}"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce t1 t2)

# ── G14_silu ARGS ─────────────────────────────────────────────
EXTRA_ARGS="
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --spatial-gating
  --gating-hidden-dim 8
  --gating-activation silu
  --gating-grad-clip 0.1
  --gating-lr-scale 2.0
"

COMMON_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-image 1.0
  --delta 0.05
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --target-diversity-var 0.06
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
"

# ── 전체 환자 ID 로드 (192 cases) ─────────────────────────────
mapfile -t ALL_IDS < <(
    tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r' | grep -v '^$'
)

echo ""
echo "================================================================"
echo "  G14_silu 전체 모달리티 재구성 스크립트"
echo "  전체 케이스: ${#ALL_IDS[@]}명 × 4 modalities = $((${#ALL_IDS[@]} * 4)) cases"
echo "  모달리티: ${MODALITIES[*]}"
echo "  출력 경로: ${RECON_ROOT}"
echo "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_EVAL=${SKIP_EVAL}"
echo "================================================================"

# ================================================================
# 단일 케이스 실행 함수
# ================================================================
FAILED_CASES=()

run_case() {
    local FULL_ID=$1
    local MODALITY=$2

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
    local OUTPUT_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] 이미 존재: ${PATIENT} / ${MODALITY}"
        return 0
    fi

    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${PATIENT}_${MODALITY}_input_missing")
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
        ${EXTRA_ARGS}

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${PATIENT}_${MODALITY}")
    else
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"
    fi
}

# ================================================================
# 학습: 48 cases × flair, t1ce, t1, t2
# ================================================================
TOTAL=0

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  학습 시작: ${EXP_TAG} / flair, t1ce, t1, t2"
    echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    for FULL_ID in "${ALL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "${FULL_ID}" "${MODALITY}"
            TOTAL=$((TOTAL + 1))
        done
    done

    SUCCESS=$(( TOTAL - ${#FAILED_CASES[@]} ))
    echo ""
    echo "  학습 전체 완료: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  성공: ${SUCCESS} / ${TOTAL}"
else
    echo "  [SKIP] 학습 (SKIP_TRAIN=1)"
fi

# ================================================================
# 평가: flair, t1ce, t1, t2 모달리티
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    echo ""
    echo "################################################################"
    echo "  평가 시작: ${EXP_TAG} / flair, t1ce, t1, t2"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    EVAL_CSV="eval_G14s_full192.csv"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs "${RECON_ROOT}:${EXP_TAG}" \
        --degraded-root "${DEGRADED_ROOT}" \
        --test-csv      "${TEST_CSV}" \
        --modalities    flair t1ce t1 t2 \
        --output-csv    "${EVAL_CSV}"

    if [ $? -eq 0 ]; then
        echo "  ✅ 평가 완료 → ${EVAL_CSV}"
    else
        echo "  ❌ 평가 실패: ${EVAL_CSV}"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 최종 요약 ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  [G14s_GA] 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
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
echo "  기준 모델  : G14_silu"
echo "  추가 모달리티: flair, t1ce, t1, t2. 게이팅 가중치 기록을 위해서 재진행"
echo "  출력 경로  : ${RECON_ROOT}"
echo "================================================================"

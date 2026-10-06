#!/usr/bin/env bash
# ================================================================
# run_G14s_no_inter.sh
# R2-3 대응: interleaved artifact 없는 조건에서 FRAME(풀 모델) + w/o Gating만 실행
#
# 인자 출처:
#   - 풀 모델(G14s)   : run_G14s_GA.sh의 EXTRA_ARGS 그대로
#   - w/o Gating      : run_G14s_abl.sh의 G14s_no_gating 조건 그대로
#   - COMMON_ARGS     : run_G14s_abl.sh COMMON_ARGS 기준 통일
#     (ff-alpha, gating-grad-clip, gating-lr-scale을 공통으로 두고
#      두 세트 모두에서 참조하도록 함 — no_gating에서는 게이팅이
#      꺼져 있으므로 gating-* 인자는 무해)
#
# 변경점 (원본 대비):
#   - DEGRADED_ROOT: interleaved 없는 BraTS 열화셋
#   - RECON_ROOT: 새 태그로 분리 (기존 4x+interleaved 결과 덮어쓰기 방지)
#   - MODALITIES: flair, t1ce만 (원 GA 스크립트는 4개 모달리티였음)
#   - set -e 제거: 개별 케이스 실패가 전체 스크립트를 죽이지 않도록
# ================================================================

set -uo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s_no_inter"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_no_inter"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── 공통 인자 (run_G14s_abl.sh COMMON_ARGS와 동일) ────────────
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

# ── 세트별 EXTRA_ARGS ─────────────────────────────────────────
declare -A EXTRA_ARGS

# 풀 모델 (run_G14s_GA.sh EXTRA_ARGS와 동일한 게이팅 on 조건)
EXTRA_ARGS["G14s_no_inter"]="
  --weight-ff-loss 0.05
  --spatial-gating
  --gating-hidden-dim 8
  --gating-activation silu
"

# w/o Gating (run_G14s_abl.sh G14s_no_gating과 동일)
EXTRA_ARGS["G14s_no_gating_no_inter"]="
  --weight-ff-loss 0.05
  --no-gating
"

RUN_SETS=("G14s_no_inter" "G14s_no_gating_no_inter")

# ── 환자 ID 로드 (96 cases) ───────────────────────────────────
mapfile -t ALL_IDS < <(
    tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r' | grep -v '^$'
)

echo ""
echo "================================================================"
echo "  R2-3: interleaved artifact 제거 조건 (FRAME + w/o Gating)"
echo "  전체 케이스: ${#ALL_IDS[@]}명 × ${#MODALITIES[@]} modalities"
echo "  실행 세트: ${RUN_SETS[*]}"
echo "  열화 루트: ${DEGRADED_ROOT}"
echo "  출력 루트: ${RECON_ROOT}"
echo "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_EVAL=${SKIP_EVAL}"
echo "================================================================"

FAILED_CASES=()
TOTAL=0

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
        echo "  ❌ 입력 파일 없음: ${INPUT}"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}_input_missing")
        return 0
    fi

    echo ""
    echo "  [START] ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"

    # shellcheck disable=SC2086
    if nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --output-model  "${DST_PT}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${COMMON_ARGS} \
        ${EXTRA}
    then
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"
    else
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    fi
}

# ── 학습: 두 세트 순차 실행 ────────────────────────────────────
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

        echo "  [완료] ${EXP_TAG}  $(date '+%Y-%m-%d %H:%M:%S')"
    done

    SUCCESS=$(( TOTAL - ${#FAILED_CASES[@]} ))
    echo ""
    echo "  학습 전체 완료: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  성공: ${SUCCESS} / ${TOTAL}"
else
    echo "  [SKIP] 학습 (SKIP_TRAIN=1)"
fi

# ── 평가: 이 두 세트만 비교 ─────────────────────────────────────
if [ "${SKIP_EVAL}" -eq 0 ]; then
    EVAL_CSV="eval_G14s_no_inter_2sets.csv"

    if /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${RECON_ROOT}/G14s_no_inter:Ours" \
            "${RECON_ROOT}/G14s_no_gating_no_inter:no_gating" \
        --degraded-root "${DEGRADED_ROOT}" \
        --test-csv      "${TEST_CSV}" \
        --modalities    "${MODALITIES[@]}" \
        --output-csv    "${EVAL_CSV}"
    then
        echo "  ✅ 평가 완료 → ${EVAL_CSV}"
    else
        echo "  ❌ 평가 실패"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

echo ""
echo "================================================================"
echo "  실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do echo "     - ${c}"; done
fi
echo "================================================================"
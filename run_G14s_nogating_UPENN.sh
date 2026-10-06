#!/usr/bin/env bash
# ================================================================
# run_G14s_nogating_UPENN.sh
# R1-2 대응: UPENN-GBM에 대한 w/o Gating 재구성
#
# BraTS ablation(run_G14s_abl.sh)의 G14s_no_gating 조건을 그대로 재현:
#   EXTRA: --weight-ff-loss 0.05 --no-gating
#   COMMON_ARGS는 원본과 동일 (gating-lr-scale/grad-clip 포함)
# UPENN용 수정: 파일명 규칙, ID 식별자, CSV 파싱, INTENSITY_MEAN=290
# ================================================================

set -uo pipefail   # -e 제거: 실패 케이스를 기록하고 계속 진행

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 ─────────────────────────────────────────────────────
EXP_TAG="G14s_no_gating"
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s_UPENN_abl"
FULL_MODEL_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s_UPENN"   # 비교용 (Ours)
DEGRADED_ROOT="/data/UPENN-GBM_pre"
GT_ROOT="/data/UPENN-GBM/UPENN-GBM_TrainingData"
TEST_CSV="/data/UPENN-GBM_pre/upenn_gbm_ids_with_segm_and_survival.csv"
ID_COLUMN_NAME="Brats20ID"

INTENSITY_MEAN=290
MODALITIES=(flair t1ce)
declare -A MODALITY_FILE_SUFFIX=( [flair]="FLAIR" [t1ce]="T1GD" )

# ── BraTS ablation과 동일한 COMMON_ARGS ──────────────────────
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

# ── G14s_no_gating 전용 인자 (BraTS ablation과 동일) ─────────
EXTRA="
  --weight-ff-loss 0.05
  --no-gating
"

# ── CSV에서 ID 추출 + 실제 폴더 존재하는 환자만 사용 ──────────
mapfile -t RAW_IDS < <(
    awk -F',' -v col="${ID_COLUMN_NAME}" '
        NR==1 { for (i=1;i<=NF;i++) { gsub(/[" \r]/,"",$i); if ($i==col) target=i } }
        NR>1  { gsub(/[" \r]/,"",$target); print $target }
    ' "${TEST_CSV}" | sed '/^$/d'
)

ALL_IDS=()
for id in "${RAW_IDS[@]}"; do
    if [ -d "${DEGRADED_ROOT}/${id}" ]; then
        ALL_IDS+=("${id}")
    else
        echo "  ⚠️  폴더 없음, 건너뜀: ${id}"
    fi
done

echo ""
echo "================================================================"
echo "  ${EXP_TAG} (UPENN-GBM, R1-2)"
echo "  CSV 환자 수: ${#RAW_IDS[@]} / 실제 사용: ${#ALL_IDS[@]} × ${#MODALITIES[@]} modalities"
echo "  출력 경로: ${RECON_ROOT}/${EXP_TAG}"
echo "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_EVAL=${SKIP_EVAL}"
echo "================================================================"

FAILED_CASES=()
TOTAL=0

run_case() {
    local FULL_ID=$1
    local MODALITY=$2
    local FILE_SUFFIX="${MODALITY_FILE_SUFFIX[$MODALITY]}"

    local PATIENT_DIR="${RECON_ROOT}/${EXP_TAG}/${FULL_ID}"
    local OUTPUT_VOL="${PATIENT_DIR}/${FULL_ID}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local DST_PT="${PATIENT_DIR}/${FULL_ID}_${MODALITY}_4x5_${EXP_TAG}.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${FILE_SUFFIX}.nii.gz"

    mkdir -p "${PATIENT_DIR}"

    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] 이미 존재: ${FULL_ID} / ${MODALITY}"
        return 0
    fi
    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${INPUT}"
        FAILED_CASES+=("${FULL_ID}_${MODALITY}_input_missing")
        return 0
    fi

    echo ""
    echo "  [START] ${EXP_TAG} / ${FULL_ID} / ${MODALITY}  $(date '+%H:%M:%S')"

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
        echo "  ✅ DONE : ${FULL_ID} / ${MODALITY}  $(date '+%H:%M:%S')"
    else
        echo "  ❌ FAILED: ${FULL_ID} / ${MODALITY}"
        FAILED_CASES+=("${FULL_ID}_${MODALITY}")
    fi
}

# ── 학습 ─────────────────────────────────────────────────────
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo "  학습 시작: $(date '+%Y-%m-%d %H:%M:%S')"
    for FULL_ID in "${ALL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "${FULL_ID}" "${MODALITY}"
            TOTAL=$((TOTAL + 1))
        done
    done
    echo "  학습 완료: $(date '+%Y-%m-%d %H:%M:%S')  (실패 ${#FAILED_CASES[@]} / ${TOTAL})"
else
    echo "  [SKIP] 학습 (SKIP_TRAIN=1)"
fi

# ── 평가: Ours(풀 모델)와 no_gating을 같은 CSV에 ─────────────
if [ "${SKIP_EVAL}" -eq 0 ]; then
    EVAL_CSV="eval_${EXP_TAG}_UPENN.csv"

    if /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_upenn.py \
        --recon-dirs "${FULL_MODEL_ROOT}:Ours" "${RECON_ROOT}/${EXP_TAG}:no_gating" \
        --gt-root    "${GT_ROOT}" \
        --test-csv   "${TEST_CSV}" \
        --modalities "${MODALITIES[@]}" \
        --output-csv "${EVAL_CSV}"
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
echo "  [${EXP_TAG}] 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do echo "     - ${c}"; done
fi
echo "================================================================"
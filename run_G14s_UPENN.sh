#!/usr/bin/env bash
# ================================================================
# run_G14s_UPENN.sh  (v1)
# R1-2 대응: UPENN-GBM 데이터셋에 대한 FRAME(G14s) 재구성
#
# BraTS용 run_G14s_GA.sh를 기반으로 수정:
#   - 모달리티: flair, t1ce만 (Table 1과 비교 가능하도록)
#   - 파일명 규칙: UPENN 대문자 모달리티명(FLAIR, T1GD), .nii.gz
#   - PATIENT 식별자: FULL_ID를 그대로 사용 (BraTS식 접미사 추출 제거)
#   - CSV 파싱: 헤더에서 'Brats20ID' 컬럼 위치를 찾아서 추출
#   - 폴더 존재 여부로 누락 환자(430번 등) 자동 스킵
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
EXP_TAG="G14s_UPENN"
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_${EXP_TAG}"
DEGRADED_ROOT="/data/UPENN-GBM_pre"
TEST_CSV="/data/UPENN-GBM_pre/upenn_gbm_ids_with_segm_and_survival.csv"
ID_COLUMN_NAME="Brats20ID"

# UPENN 원본(전처리 후) 볼륨의 non-zero voxel mean intensity를 확인한 뒤 290으로 수정 완료
INTENSITY_MEAN=290   

# R1-2는 Table 1과 비교 가능해야 하므로 flair/t1ce만 사용
MODALITIES=(flair t1ce)
declare -A MODALITY_FILE_SUFFIX=( [flair]="FLAIR" [t1ce]="T1GD" )

# ── G14_silu ARGS (BraTS 실험과 동일한 하이퍼파라미터 고정) ────
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

# ── CSV에서 'Brats20ID' 컬럼을 이름으로 찾아 ID 추출 ─────────
mapfile -t RAW_IDS < <(
    awk -F',' -v col="${ID_COLUMN_NAME}" '
        NR==1 { for (i=1;i<=NF;i++) if ($i==col) target=i }
        NR>1  { print $target }
    ' "${TEST_CSV}" | tr -d '\r' | sed '/^$/d'
)

# ── 실제 폴더가 존재하는 환자만 남김 (누락 환자 자동 스킵) ────
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
echo "  ${EXP_TAG}  UPENN-GBM 재구성 스크립트 (R1-2)"
echo "  CSV 원본 환자 수 : ${#RAW_IDS[@]}"
echo "  실제 사용 환자 수: ${#ALL_IDS[@]} × ${#MODALITIES[@]} modalities"
echo "  출력 경로: ${RECON_ROOT}"
echo "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_EVAL=${SKIP_EVAL}"
echo "================================================================"

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ================================================================
# 단일 케이스 실행 함수
# ================================================================
run_case() {
    local FULL_ID=$1
    local MODALITY=$2
    local FILE_SUFFIX="${MODALITY_FILE_SUFFIX[$MODALITY]}"

    local PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
    local OUTPUT_VOL="${PATIENT_DIR}/${FULL_ID}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local DST_PT="${PATIENT_DIR}/${FULL_ID}_${MODALITY}_4x5_${EXP_TAG}.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${FILE_SUFFIX}.nii.gz"

    mkdir -p "${PATIENT_DIR}"

    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] 이미 존재: ${FULL_ID} / ${MODALITY}"
        return 0
    fi

    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${FULL_ID} / ${MODALITY} (${INPUT})"
        FAILED_CASES+=("${FULL_ID}_${MODALITY}_input_missing")
        return 0
    fi

    echo ""
    echo "  [START] ${EXP_TAG} / ${FULL_ID} / ${MODALITY}  $(date '+%H:%M:%S')"

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
        echo "  ❌ FAILED: ${EXP_TAG} / ${FULL_ID} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${FULL_ID}_${MODALITY}")
    else
        echo "  ✅ DONE : ${EXP_TAG} / ${FULL_ID} / ${MODALITY}  $(date '+%H:%M:%S')"
    fi
}

# ================================================================
# 학습: N cases × flair, t1ce
# ================================================================
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  학습 시작: ${EXP_TAG} / ${MODALITIES[*]}"
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
# 평가
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  평가 시작: ${EXP_TAG} / ${MODALITIES[*]}"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    EVAL_CSV="eval_${EXP_TAG}.csv"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_upenn.py \
        --recon-dirs "${RECON_ROOT}:Ours" \
        --gt-root    "/data/UPENN-GBM/UPENN-GBM_TrainingData" \
        --test-csv   "${TEST_CSV}" \
        --modalities "${MODALITIES[@]}" \
        --output-csv "${EVAL_CSV}"

    if [ $? -eq 0 ]; then
        echo "  ✅ 평가 완료 → ${EVAL_CSV}"
    else
        echo "  ❌ 평가 실패: ${EVAL_CSV}"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

echo ""
echo "================================================================"
echo "  [${EXP_TAG}] 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  성공: ${SUCCESS:-0} / ${TOTAL:-0}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo "================================================================"
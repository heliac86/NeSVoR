#!/usr/bin/env bash
# ================================================================
# run_G14s_scale.sh  (v1)
# FRAME(G14_silu)을 2x / 6x 열화 데이터에 적용, 볼륨(.nii.gz) + 모델(.pt, 게이팅 가중치 포함) 저장
#
# 사용 예:
#   FACTOR=2 ./run_G14s_scale.sh
#   FACTOR=6 ./run_G14s_scale.sh
#   FACTOR=6 SKIP_TRAIN=1 ./run_G14s_scale.sh        # 평가만
#   FACTOR=2 MODALITIES_STR="flair t1ce t1 t2" ./run_G14s_scale.sh
#
# 출력:
#   recon_G14s_{F}x/BraTS20_Training_{ID}/{ID}_{mod}_{F}x_G14s.{nii.gz,pt}
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 배율별 설정 (σ = 0.375 × factor 규칙) ────────────────────
FACTOR=${FACTOR:?"FACTOR=2 또는 FACTOR=6 으로 지정하세요"}
case "${FACTOR}" in
    2) DEGRADED_ROOT="/data/BraTS20_Degraded_2x_sig075" ;;
    6) DEGRADED_ROOT="/data/BraTS20_Degraded_6x_sig225" ;;
    *) echo "지원하지 않는 FACTOR: ${FACTOR}"; exit 1 ;;
esac

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

EXP_TAG="G14s"
PY="/dshome/ddualab/conda_env/nerf_tcnn/bin/python"
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_${EXP_TAG}_${FACTOR}x"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
INTENSITY_MEAN=308            # 4x 실험과 동일하게 유지

MODALITIES_STR=${MODALITIES_STR:-"flair t1ce"}
read -ra MODALITIES <<< "${MODALITIES_STR}"

# ── G14_silu ARGS (4x와 동일, 배율별 재튜닝 없음) ─────────────
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

mapfile -t ALL_IDS < <(
    tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r' | grep -v '^$'
)

echo "================================================================"
echo "  ${EXP_TAG} / ${FACTOR}x 재구성"
echo "  입력: ${DEGRADED_ROOT}"
echo "  출력: ${RECON_ROOT}"
echo "  환자 ${#ALL_IDS[@]}명 × 모달리티 (${MODALITIES[*]})"
echo "================================================================"

# ── 안전장치: 입력 헤더의 z 간격이 FACTOR와 일치하는지 확인 ──────
CHECK_FILE="${DEGRADED_ROOT}/${ALL_IDS[0]}/${ALL_IDS[0]}_${MODALITIES[0]}.nii"
"${PY}" - <<EOF
import nibabel as nib, sys
z = nib.load("${CHECK_FILE}").header.get_zooms()[2]
print(f"[CHECK] z spacing = {z:.2f} (expected ${FACTOR})")
sys.exit(0 if abs(z - ${FACTOR}) < 1e-3 else 1)
EOF

FAILED_CASES=()
TOTAL=0

run_case() {
    local FULL_ID=$1
    local MODALITY=$2
    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
    local STEM="${PATIENT_DIR}/${PATIENT}_${MODALITY}_${FACTOR}x_${EXP_TAG}"
    local OUTPUT_VOL="${STEM}.nii.gz"
    local DST_PT="${STEM}.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    # 볼륨과 .pt가 둘 다 있어야 완료로 간주
    if [ -f "${OUTPUT_VOL}" ] && [ -f "${DST_PT}" ]; then
        echo "  [SKIP] ${PATIENT} / ${MODALITY}"
        return 0
    fi
    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 없음: ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${PATIENT}_${MODALITY}_input_missing")
        return 0
    fi

    echo "  [START] ${FACTOR}x / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"

    # shellcheck disable=SC2086
    if nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --output-model  "${DST_PT}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${COMMON_ARGS} \
        ${EXTRA_ARGS}; then
        echo "  ✅ DONE : ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"
    else
        echo "  ❌ FAILED: ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${PATIENT}_${MODALITY}")
    fi
}

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo "학습 시작: $(date '+%Y-%m-%d %H:%M:%S')"
    for FULL_ID in "${ALL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "${FULL_ID}" "${MODALITY}"
            TOTAL=$((TOTAL + 1))
        done
    done
    echo "학습 종료: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "  [SKIP] 학습"
fi

if [ "${SKIP_EVAL}" -eq 0 ]; then
    EVAL_CSV="eval_${EXP_TAG}_${FACTOR}x.csv"
    if "${PY}" evaluate_batch_v2.py \
        --recon-dirs "${RECON_ROOT}:${EXP_TAG}" \
        --degraded-root "${DEGRADED_ROOT}" \
        --test-csv      "${TEST_CSV}" \
        --modalities    "${MODALITIES[@]}" \
        --output-csv    "${EVAL_CSV}"; then
        echo "  ✅ 평가 완료 → ${EVAL_CSV}"
    else
        echo "  ❌ 평가 실패"
    fi
else
    echo "  [SKIP] 평가"
fi

echo "================================================================"
echo "  실행 완료: $(date '+%Y-%m-%d %H:%M:%S')  (총 ${TOTAL}건)"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패 ${#FAILED_CASES[@]}건:"
    for c in "${FAILED_CASES[@]}"; do echo "     - ${c}"; done
fi
echo "================================================================"
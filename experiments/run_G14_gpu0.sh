#!/usr/bin/env bash
# ==============================================================================
# G14 실험 세트 — GPU 0 담당: G14_clamp02, G14_clamp05
# 기준: G12_h8z (spatial-gating, G7 모드, hidden=8, ff=0.05, 96 cases)
# 새 인자: --gating-min-weight (clamp 하한)
# ==============================================================================
set -euo pipefail

GPU=0
BASE_ARGS="\
    --spatial-gating \
    --gating-hidden-dim 8 \
    --weight-ff-loss 0.05 \
    --n-patches 8 \
    --patch-size 16 \
    --ff-alpha 1.0"

DATA_ROOT="${DATA_ROOT:-/data/nesvor}"
OUT_ROOT="${OUT_ROOT:-/results/G14}"
CASE_LIST="${CASE_LIST:-${DATA_ROOT}/cases.txt}"

run_case() {
    local CASE_ID="$1"
    local EXP_NAME="$2"
    local EXTRA_ARGS="$3"

    local OUT_DIR="${OUT_ROOT}/${EXP_NAME}/${CASE_ID}"
    mkdir -p "${OUT_DIR}"

    echo "[$(date '+%H:%M:%S')] START ${EXP_NAME} | case=${CASE_ID} | GPU=${GPU}"

    nesvor reconstruct \
        --input-stacks  "${DATA_ROOT}/${CASE_ID}"/*.nii.gz \
        --output-volume "${OUT_DIR}/recon.nii.gz" \
        --output-model  "${OUT_DIR}/model.pt" \
        --device        ${GPU} \
        --seed          42 \
        ${BASE_ARGS} \
        ${EXTRA_ARGS} \
        --output-json   "${OUT_DIR}/args.json" \
        2>&1 | tee "${OUT_DIR}/log.txt"

    echo "[$(date '+%H:%M:%S')] DONE  ${EXP_NAME} | case=${CASE_ID}"
}

# ------------------------------------------------------------------------------
# G14_clamp02 : --gating-min-weight 0.2
# ------------------------------------------------------------------------------
echo "======================================================"
echo " G14_clamp02  (GPU ${GPU})"
echo "======================================================"
while IFS= read -r CASE_ID; do
    [[ -z "${CASE_ID}" || "${CASE_ID}" == \#* ]] && continue
    run_case "${CASE_ID}" "G14_clamp02" "--gating-min-weight 0.2"
done < "${CASE_LIST}"

# ------------------------------------------------------------------------------
# G14_clamp05 : --gating-min-weight 0.5
# ------------------------------------------------------------------------------
echo "======================================================"
echo " G14_clamp05  (GPU ${GPU})"
echo "======================================================"
while IFS= read -r CASE_ID; do
    [[ -z "${CASE_ID}" || "${CASE_ID}" == \#* ]] && continue
    run_case "${CASE_ID}" "G14_clamp05" "--gating-min-weight 0.5"
done < "${CASE_LIST}"

echo "[GPU ${GPU}] All G14 (clamp02, clamp05) experiments finished."

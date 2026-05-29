#!/usr/bin/env bash
# ================================================================
# run_G13_gpu3.sh  (v1 — 2026-05-29)  [GPU 3번 전용]
# 세트: G13_ff012 / G13_ff015  — weight-ff-loss 탐색
#
# ── 실험 배경 ──────────────────────────────────────────────────
# GPU2와 동일한 배경. GPU3은 더 큼 ff-loss 값을 담당.
#
# ── 탐색 설계 ───────────────────────────────────────────────
# GPU2 파일:  G13_ff008 (0.08), G13_ff010 (0.10)
# 이 파일:   G13_ff012 (0.12), G13_ff015 (0.15)
#
# 0.12, 0.15를 타는 이유:
#   0.10에서 LPIPS가 회복된다면 0.12/0.15는 범위 바깥으로 쏼우는 저항과색
#   0.10에서 아직 회복 안된다면 더 강한 생음 필요
#   0.15는 FF Loss가 PSNR을 지나치게 희생하는 상한선 확인용
#
# ── 사용법 ───────────────────────────────────────────────────
#   chmod +x run_G13_gpu3.sh
#   CUDA_VISIBLE_DEVICES=3 ./run_G13_gpu3.sh
#   SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=3 ./run_G13_gpu3.sh
#   ENABLE_FF012=0 CUDA_VISIBLE_DEVICES=3 ./run_G13_gpu3.sh
#   ENABLE_FF015=0 CUDA_VISIBLE_DEVICES=3 ./run_G13_gpu3.sh
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ENABLE_FF012=${ENABLE_FF012:-1}
ENABLE_FF015=${ENABLE_FF015:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G13"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"
G12_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G12"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

mapfile -t PILOT_IDS < <(
    tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r' \
    | python3 -c "
import sys, random
ids = [l.strip() for l in sys.stdin if l.strip()]
random.seed(42)
print('\n'.join(random.sample(ids, 10)))
"
)

echo ""
echo "================================================================"
echo "  G13 FF Loss 탐색 파일럿  [GPU 3번]"
echo "  세트: G13_ff012 (0.12) / G13_ff015 (0.15)"
echo "  환자: ${#PILOT_IDS[@]}명 (seed=42)"
echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

BASE_ARGS="
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
  --spatial-gating
  --gating-hidden-dim 8
"
# --no-z-gating 없음 → G7 모드 (G12_h8z와 동일)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

run_case() {
    local EXP_TAG=$1
    local FULL_ID=$2
    local MODALITY=$3
    local FF_WEIGHT=$4

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${EXP_TAG}/${FULL_ID}"
    local OUTPUT_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        return 0
    fi
    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 없음: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}_input_missing")
        TOTAL=$((TOTAL + 1))
        return 0
    fi

    echo "  [START] ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${BASE_ARGS} \
        --weight-ff-loss "${FF_WEIGHT}"

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"
    fi
}

# ================================================================
# 학습
# ================================================================
if [ "${SKIP_TRAIN}" -eq 0 ]; then

# ── 세트 G13_ff012: weight-ff-loss=0.12 ───────────────────────────
if [ "${ENABLE_FF012}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  G13_ff012: weight-ff-loss=0.12  (+0.07 vs G12_h8z)"
    echo "  케이스: 20  예상 ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G13_ff012" "${FULL_ID}" "${MODALITY}" "0.12"
        done
    done
else
    echo "  [SKIP] G13_ff012"
fi

# ── 세트 G13_ff015: weight-ff-loss=0.15 ───────────────────────────
if [ "${ENABLE_FF015}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  G13_ff015: weight-ff-loss=0.15  (3배 vs G12_h8z, 상한 확인용)"
    echo "  케이스: 20  예상 ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G13_ff015" "${FULL_ID}" "${MODALITY}" "0.15"
        done
    done
else
    echo "  [SKIP] G13_ff015"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가: P2_LR2 / G12_h8z / G13_ff012 / G13_ff015
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    PILOT_CSV="/tmp/pilot_test_G13.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    echo ""
    echo "################################################################"
    echo "  평가: evaluate_batch_v2.py  [GPU3]"
    echo "  4세트: P2_LR2 / G12_h8z / G13_ff012 / G13_ff015"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                  \
            "${G12_RECON_ROOT}/G12_h8z:G12_h8z_ff005"       \
            "${RECON_ROOT}/G13_ff012:G13_ff012"              \
            "${RECON_ROOT}/G13_ff015:G13_ff015"              \
        --degraded-root "${DEGRADED_ROOT}"                   \
        --test-csv      "${PILOT_CSV}"                       \
        --modalities    flair t1ce                           \
        --output-csv    eval_G13_gpu3.csv

    if [ $? -eq 0 ]; then
        echo "  ✅ 평가 완료 → eval_G13_gpu3.csv"
        echo ""
        echo "  ── 판단 기준 (96-case 전체 P2_LR2 기준) ──────────────────"
        echo "  목표: FLAIR PSNR > 26.845 AND LPIPS < 0.02662"
        echo "         T1CE  PSNR > 26.373 AND LPIPS < 0.04449"
        echo "  G12_h8z(ff=0.05): FLAIR 27.177/0.02870 | T1CE 26.603/0.04577"
        echo "  ─────────────────────────────────────────────────────────"
        echo "  GPU2 결과(ff=0.08/0.10)는 eval_G13_gpu2.csv 참조"
        echo ""
        echo "  ── ff-loss 켜비운 방향 가이드 ────────────────────────"
        echo "  LPIPS 단조 감소 AND PSNR 유지 → 해당 ff값으로 full run"
        echo "  LPIPS 감소 없음 → ff-loss 효과 없음, SiLU 탐색"
        echo "  PSNR 하락 시작 → ff-loss 과자, 0.05~0.10 중간값 탐색"
        echo "  ─────────────────────────────────────────────────────────"
    else
        echo "  ❌ 평가 실패"
    fi
else
    echo "  [SKIP] 평가"
fi

echo ""
echo "================================================================"
echo "  [GPU3] 완료: $(date '+%Y-%m-%d %H:%M:%S')  성공 ${SUCCESS}/${TOTAL}"
if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    for c in "${FAILED_CASES[@]}"; do echo "  ❌ ${c}"; done
fi
echo "  결과 CSV: eval_G13_gpu3.csv"
echo "  G13_ff012: weight-ff-loss=0.12, G7 모드"
echo "  G13_ff015: weight-ff-loss=0.15, G7 모드"
echo "================================================================"

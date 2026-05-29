#!/usr/bin/env bash
# ================================================================
# run_G13_gpu2.sh  (v1 — 2026-05-29)  [GPU 2번 전용]
# 세트: G13_ff008 / G13_ff010  — weight-ff-loss 탐색
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G12_h8z (96 cases, G7 모드):
#   FLAIR PSNR 27.177 (+0.332 vs P2_LR2) ✅ 양쪽 PSNR 우위
#   FLAIR LPIPS 0.02870 (+0.00208 vs P2_LR2) ❌ LPIPS 악화
#   T1CE  PSNR 26.603 (+0.230 vs P2_LR2) ✅
#   T1CE  LPIPS 0.04577 (+0.00128 vs P2_LR2) ❌
#
# 원인 추론:
#   z_repr이 GatingMLP에 입력되어 특정 주파수 성분 과도 강조 가능성.
#   FF Loss는 주파수 도메인에서 고주파 성분을 복원.
#   weight-ff-loss를 높이면 고주파 표현이 강화되어
#   LPIPS 력회복 가능성.
#
# ── 탐색 설계 ───────────────────────────────────────────────
# 이 파일:  G13_ff008 (0.08), G13_ff010 (0.10)
# GPU3 파일: G13_ff012 (0.12), G13_ff015 (0.15)
# 기데: weight-ff-loss 단조 증가로 LPIPS 방향성 파악.
#
# 탐색 범위 근거:
#   0.05 (G12_h8z): LPIPS 0.02870 (P2_LR2 대비 악화)
#   0.08: 소폭 증가, PSNR 유지 가능성 높음
#   0.10: G1 P2_LR2 세팅(ff=0.05)의 2배. 종종 장벽 효과.
#   0.12, 0.15: LPIPS를 더 크게 좌우하는지 확인.
#
# ── 판단 기준 ───────────────────────────────────────────────
# 목표: PSNR AND LPIPS 양쪽 P2_LR2 이상
#   P2_LR2 기준 (96 cases):
#     FLAIR PSNR 26.845 / LPIPS 0.02662
#     T1CE  PSNR 26.373 / LPIPS 0.04449
#   G12_h8z 기준 (96 cases):
#     FLAIR PSNR 27.177 / LPIPS 0.02870
#     T1CE  PSNR 26.603 / LPIPS 0.04577
#
# ── 비교 기준점 (96-case 전체) ──────────────────────────────
#   no_gating: FLAIR 26.643/0.0266 / T1CE 26.332/0.0445
#   P2_LR2:    FLAIR 26.845/0.0266 / T1CE 26.373/0.0445
#
# ── 스케일 ────────────────────────────────────────────────────
# 파일럿 10명 × 2 modalities × 2 세트 = 40 cases
# 예상 시간: ~1.5시간/세트 × 2 ≈ ~3시간
#
# ── 사용법 ───────────────────────────────────────────────────
#   chmod +x run_G13_gpu2.sh
#   CUDA_VISIBLE_DEVICES=2 ./run_G13_gpu2.sh
#   SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=2 ./run_G13_gpu2.sh
#   ENABLE_FF008=0 CUDA_VISIBLE_DEVICES=2 ./run_G13_gpu2.sh
#   ENABLE_FF010=0 CUDA_VISIBLE_DEVICES=2 ./run_G13_gpu2.sh
#
# ── 출력 파일 명명 규칙 ──────────────────────────────────────
#   recon_G13/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ENABLE_FF008=${ENABLE_FF008:-1}
ENABLE_FF010=${ENABLE_FF010:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G13"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"
G12_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G12"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── 파일럿 환자 10명 (seed=42, 이전 실험과 동일) ─────────────
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
echo "  G13 FF Loss 탐색 파일럿  [GPU 2번]"
echo "  세트: G13_ff008 (0.08) / G13_ff010 (0.10)"
echo "  환자: ${#PILOT_IDS[@]}명 (seed=42)"
echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ── BASE_ARGS: G12_h8z 완전 재현 (ff-loss만 변경) ───────────────
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
# weight-ff-loss는 세트별로 각각 지정

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

# ── 세트 G13_ff008: weight-ff-loss=0.08 ───────────────────────────
if [ "${ENABLE_FF008}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  G13_ff008: weight-ff-loss=0.08  (+0.03 vs G12_h8z)"
    echo "  케이스: 20  예상 ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G13_ff008" "${FULL_ID}" "${MODALITY}" "0.08"
        done
    done
else
    echo "  [SKIP] G13_ff008"
fi

# ── 세트 G13_ff010: weight-ff-loss=0.10 ───────────────────────────
if [ "${ENABLE_FF010}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  G13_ff010: weight-ff-loss=0.10  (2배 vs G12_h8z)"
    echo "  케이스: 20  예상 ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G13_ff010" "${FULL_ID}" "${MODALITY}" "0.10"
        done
    done
else
    echo "  [SKIP] G13_ff010"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가: P2_LR2 / G12_h8z(기준) / G13_ff008 / G13_ff010
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    PILOT_CSV="/tmp/pilot_test_G13.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    echo ""
    echo "################################################################"
    echo "  평가: evaluate_batch_v2.py  [GPU2]"
    echo "  4세트: P2_LR2 / G12_h8z / G13_ff008 / G13_ff010"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                  \
            "${G12_RECON_ROOT}/G12_h8z:G12_h8z_ff005"       \
            "${RECON_ROOT}/G13_ff008:G13_ff008"              \
            "${RECON_ROOT}/G13_ff010:G13_ff010"              \
        --degraded-root "${DEGRADED_ROOT}"                   \
        --test-csv      "${PILOT_CSV}"                       \
        --modalities    flair t1ce                           \
        --output-csv    eval_G13_gpu2.csv

    if [ $? -eq 0 ]; then
        echo "  ✅ 평가 완료 → eval_G13_gpu2.csv"
        echo ""
        echo "  ── 판단 기준 (96-case 전체 P2_LR2 기준) ──────────────────"
        echo "  목표: FLAIR PSNR > 26.845 AND LPIPS < 0.02662"
        echo "         T1CE  PSNR > 26.373 AND LPIPS < 0.04449"
        echo "  G12_h8z(ff=0.05): FLAIR 27.177/0.02870 | T1CE 26.603/0.04577"
        echo "  ─────────────────────────────────────────────────────────"
        echo "  GPU3 결과(ff=0.12/0.15)는 eval_G13_gpu3.csv 참조"
    else
        echo "  ❌ 평가 실패"
    fi
else
    echo "  [SKIP] 평가"
fi

echo ""
echo "================================================================"
echo "  [GPU2] 완료: $(date '+%Y-%m-%d %H:%M:%S')  성공 ${SUCCESS}/${TOTAL}"
if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    for c in "${FAILED_CASES[@]}"; do echo "  ❌ ${c}"; done
fi
echo "  결과 CSV: eval_G13_gpu2.csv"
echo "  G13_ff008: weight-ff-loss=0.08, G7 모드"
echo "  G13_ff010: weight-ff-loss=0.10, G7 모드"
echo "================================================================"

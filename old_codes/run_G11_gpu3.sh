#!/usr/bin/env bash
# ================================================================
# run_G11_gpu3.sh  (v1 — 2026-05-28)  [GPU 3번 전용]
# 세트 C: P2_LR2_2500 — 전역 gating (G1), n-iter=2500
# 평가: P2_LR2(기준, 2000) vs P2_LR2_2500 → iter 증가가 G1에 미치는 효과
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G9 full run에서 P2_LR2(n-iter=2000)가 현재 최고:
#   FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB (96-case)
#
# GPU2(run_G11_gpu2.sh)에서 G11_h16_2500(spatial, n-iter=2500)을 실행 중.
# 이 파일에서는 G1 전역 gating도 동일하게 n-iter=2500으로 올려
# "iter 증가" 효과를 순수하게 분리함.
#
# ── 비교 목적 ──────────────────────────────────────────────────
# Q1. n-iter 증가 자체가 G1(전역)에도 도움이 되는가?
#     → P2_LR2(2000) vs P2_LR2_2500
# Q2. 동일 iter(2500) 조건에서 spatial이 전역을 이기는가?
#     → P2_LR2_2500 vs G11_h16_2500(GPU2 결과)
#     이 비교가 이번 G11의 핵심 질문.
#
# ── GPU 배정 ──────────────────────────────────────────────────
#   이 파일: GPU 3번
#   CUDA_VISIBLE_DEVICES=3 ./run_G11_gpu3.sh
#
# ── 총 실행 ──────────────────────────────────────────────────
#   세트 C: 20 cases (~1.9시간, n-iter 25% 증가)
#
# ── 사용법 ───────────────────────────────────────────────────
#   chmod +x run_G11_gpu3.sh
#   CUDA_VISIBLE_DEVICES=3 ./run_G11_gpu3.sh
#   SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=3 ./run_G11_gpu3.sh  # 평가만
#
# ── 출력 파일 명명 규칙 ──────────────────────────────────────
#   recon_G11/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G11"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# P2_LR2 기준: recon_G9_full/P2_LR2 재사용 (n-iter=2000)
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── seed=42 기반 파일럿 환자 10명 (G8/G10/G11_gpu2와 동일) ──
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
echo "  파일럿 환자 목록 (seed=42, 10명 — G8/G10/G11_gpu2와 동일):"
for ID in "${PILOT_IDS[@]}"; do
    echo "    - ${ID}"
done
echo "================================================================"

# ── BASE_ARGS: P2_LR2 완전 재현 (spatial-gating 없음) ────────
# G1 전역 gating 설정. n-iter만 2500으로 변경.
BASE_ARGS_G1="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-image 1.0
  --delta 0.05
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --target-diversity-var 0.06
  --gating-grad-clip 0.1
  --weight-diversity-loss 0.08
  --gating-lr-scale 2.0
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
  --n-iter 2500
"
# 주: --spatial-gating 없음 → G1 전역 level_weights 사용
#     P2_LR2와 동일한 설정에서 n-iter=2500만 변경

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ────────────────────────────────────
run_case() {
    local EXP_TAG=$1
    local FULL_ID=$2
    local MODALITY=$3
    shift 3
    local EXTRA_ARGS="$*"

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${EXP_TAG}/${FULL_ID}"
    local OUTPUT_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] 이미 존재: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        return 0
    fi

    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}_input_missing")
        TOTAL=$((TOTAL + 1))
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "  [START] Tag=${EXP_TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        echo "  Vol   → ${OUTPUT_VOL}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# 학습 실행
# ================================================================
if [ "${SKIP_TRAIN}" -eq 0 ]; then

# ================================================================
# 세트 C: P2_LR2_2500  [전역 G1 gating, n-iter=2500]
#
#   목적: P2_LR2(n-iter=2000)에서 iter만 늘렸을 때 G1 자체의
#         성능 향상 가능성 확인. 두 가지 역할:
#         1) n-iter 증가가 G1에 유익한지 (P2_LR2_2500 vs P2_LR2)
#         2) GPU2의 G11_h16_2500과 동일 iter 공정 비교 기준점
#            (P2_LR2_2500 vs G11_h16_2500)
#   변경: --n-iter 2500  (P2_LR2 대비)
#   비교: P2_LR2(2000) vs P2_LR2_2500 → iter 증가 효과 분리
# ================================================================
echo ""
echo "################################################################"
echo "  세트 C: P2_LR2_2500 — G1 전역 gating, n-iter=2500"
echo "  비교: P2_LR2(2000) vs P2_LR2_2500 → iter 증가 효과"
echo "        P2_LR2_2500 vs G11_h16_2500(GPU2) → 동일 iter, spatial 유효성"
echo "  케이스: 20  예상 시간: ~1.9시간"
echo "################################################################"
for FULL_ID in "${PILOT_IDS[@]}"; do
    for MODALITY in "${MODALITIES[@]}"; do
        run_case "P2_LR2_2500" "${FULL_ID}" "${MODALITY}" \
            ${BASE_ARGS_G1}
    done
done

fi  # end SKIP_TRAIN

# ================================================================
# 평가 실행
# P2_LR2(2000): 기존 결과 재사용 (파일럿 10명 필터링)
# P2_LR2_2500: 이번 실험 결과
# G11_h16_2500: GPU2 결과 경로 참조 (GPU2 완료 후 실행 권장)
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    PILOT_CSV="/tmp/pilot_test_G11.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    echo ""
    echo "################################################################"
    echo "  평가 시작: evaluate_batch_v2.py  [GPU3 세트]"
    echo "  총 3세트: P2_LR2(기준, 2000) / P2_LR2_2500 / G11_h16_2500"
    echo "  ※ G11_h16_2500은 GPU2 완료 후 경로 접근 가능"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                              \
            "${RECON_ROOT}/P2_LR2_2500:P2_LR2_2500"                   \
            "${RECON_ROOT}/G11_h16_2500:G11_h16_2500"                  \
        --degraded-root "${DEGRADED_ROOT}"                              \
        --test-csv      "${PILOT_CSV}"                                  \
        --modalities    flair t1ce                                      \
        --output-csv    eval_G11_gpu3.csv

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
        echo "  ※ G11_h16_2500이 아직 미완료라면 GPU2 완료 후 재실행:"
        echo "     SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=3 ./run_G11_gpu3.sh"
    else
        echo "  ✅ 평가 완료 → eval_G11_gpu3.csv"
        echo ""
        echo "  ── 비교 기준점 (96-case 전체 기준) ──────────────────────"
        echo "  no_gating:   FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB"
        echo "  P2_LR2(2000): FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB"
        echo "  ────────────────────────────────────────────────────────"
        echo "  ── GPU2 결과(eval_G11_gpu2.csv)와 함께 종합 판단 ────────"
        echo "  핵심 비교: P2_LR2_2500 vs G11_h16_2500"
        echo "             (동일 iter=2500, 전역 vs spatial 순수 비교)"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 최종 요약 ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  [GPU3] 전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패한 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo ""
echo "  ── 실험 메모 ─────────────────────────────────────────────"
echo "  기준 브랜치   : backup_IC"
echo "  P2_LR2 경로   : ${P2_LR2_RECON_ROOT}"
echo "  실험 경로     : ${RECON_ROOT}"
echo "  결과 CSV      : eval_G11_gpu3.csv"
echo "  P2_LR2_2500   : G1 전역 gating, n-iter=2500"
echo "  GPU2 연계     : G11_h16_2500과 동일 iter 비교 → eval_G11_gpu3.csv"
echo "================================================================"

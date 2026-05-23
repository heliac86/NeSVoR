#!/usr/bin/env bash
# ================================================================
# run_gating_sweep.sh  (v1 — 2026-05-21)
# Ours (IC no_hsm) 기반 gating 하이퍼파라미터 탐색
#
# ── 실험 배경 ──────────────────────────────────────────────────
# 기준점 (Ours):
#   --weight-diversity-loss 0.08
#   --target-diversity-var  0.06
#   --gating-grad-clip      0.0   (비활성화)
#   FLAIR PSNR 26.7494  (48명 × flair/t1ce = 96 cases 기준)
#
# 과거 탐색(15 cases, HSM 있는 조건)에서는 0.08/0.06이 최적이었으나,
# HSM 제거 + 96 cases 조건에서 다른 최적점이 존재할 가능성이 있음.
# diversity loss의 두 인자 사이에 교호작용이 있으므로 축 분리 탐색.
#
# ── 탐색 설계 ──────────────────────────────────────────────────
# 축 1: weight-diversity-loss (div-w)  — 0.08 → 0.12 → 0.16
# 축 2: target-diversity-var  (div-v)  — 0.06 → 0.09
# 축 3: gating-grad-clip      (g-clip) — 0.0  → 0.1 / 0.5
#
# 세트명        div-w   div-v   g-clip   목적
# ---------     -----   -----   ------   ----------------------
# Ours(기준)    0.08    0.06    0.0      기준 재사용 (재실행 없음)
# G-DH          0.12    0.06    0.0      div-w 단독 증가 효과
# G-DHH         0.16    0.06    0.0      div-w 더 강하게 (saturation 확인)
# G-DV          0.08    0.09    0.0      div-v 단독 증가 효과
# G-DHV         0.12    0.09    0.0      div-w + div-v 교호작용
# G-C2          0.08    0.06    0.1      grad-clip 강한 제약 (자주 발동)
# G-C           0.08    0.06    0.5      grad-clip 약한 제약 (드물게 발동)
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q1. div-w 증가가 Ours를 개선하는가?         → Ours vs G-DH
# Q2. div-w 강화의 포화점은 어디인가?         → G-DH vs G-DHH
# Q3. div-v 증가가 독립적으로 유효한가?       → Ours vs G-DV
# Q4. div-w + div-v 조합이 가산적인가?        → G-DH/G-DV vs G-DHV
# Q5. gating-grad-clip 활성화가 유효한가?     → Ours vs G-C2 vs G-C
# Q6. clip 값의 방향성은?                     → G-C2(0.1) vs G-C(0.5)
#
# ── 총 실행 ────────────────────────────────────────────────────
# 신규 6세트 × 96 cases = 576 케이스
# Ours(기준) 포함 7세트 비교
# 예상 총 시간: 신규 6세트 × ~7시간 ≈ 42시간 (모레 아침 결과)
#
# ── 데이터 ─────────────────────────────────────────────────────
# 환자: test.csv 기반 48명 전체
# 모달리티: flair, t1ce
# 총 케이스: 48명 × 2모달 = 96 cases / 세트
#
# ── 사용법 ──────────────────────────────────────────────────────
#   chmod +x run_gating_sweep.sh
#   ./run_gating_sweep.sh                # 학습 + 평가 전체
#   SKIP_TRAIN=1 ./run_gating_sweep.sh   # 평가만 재실행
#   SKIP_EVAL=1  ./run_gating_sweep.sh   # 학습만 실행
#   ENABLE_G_DH=0   ./run_gating_sweep.sh  # G-DH  건너뜀
#   ENABLE_G_DHH=0  ./run_gating_sweep.sh  # G-DHH 건너뜀
#   ENABLE_G_DV=0   ./run_gating_sweep.sh  # G-DV  건너뜀
#   ENABLE_G_DHV=0  ./run_gating_sweep.sh  # G-DHV 건너뜀
#   ENABLE_G_C2=0   ./run_gating_sweep.sh  # G-C2  건너뜀
#   ENABLE_G_C=0    ./run_gating_sweep.sh  # G-C   건너뜀
#
# ── 출력 파일 명명 규칙 ──────────────────────────────────────
#   recon_gating_sweep/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ──────────────────────────────────────
ENABLE_G_DH=${ENABLE_G_DH:-1}
ENABLE_G_DHH=${ENABLE_G_DHH:-1}
ENABLE_G_DV=${ENABLE_G_DV:-1}
ENABLE_G_DHV=${ENABLE_G_DHV:-1}
ENABLE_G_C2=${ENABLE_G_C2:-1}
ENABLE_G_C=${ENABLE_G_C:-1}

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_gating_sweep"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
OURS_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_IC"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── BASE_ARGS: Ours(IC no_hsm) 확정 설정 완전 재현 ──────────
# div-w, div-v, g-clip 만 세트별로 덮어씀
BASE_ARGS="
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
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
"

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── test.csv에서 환자 ID 로드 ────────────────────────────────
mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

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

    # 이미 출력 파일이 존재하면 스킵
    if [ -f "${OUTPUT_VOL}" ]; then
        echo "  [SKIP] 이미 존재: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        return 0
    fi

    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        echo "     찾는 경로: ${INPUT}"
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
# 세트 G-DH: weight-diversity-loss=0.12, target-diversity-var=0.06
#
#   목적: div-w 단독 증가 효과 확인.
#         Ours(0.08)와 div-v/g-clip 완전 동일, div-w만 증가.
#         gating 레벨 분화 압력을 높여 Ours 개선 여부 탐색.
#   변경: --weight-diversity-loss 0.12
# ================================================================
if [ "${ENABLE_G_DH}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G-DH: div-w=0.12, div-v=0.06, g-clip=0.0"
    echo "  비교: Ours (div-w=0.08) vs G-DH → div-w 단독 증가 효과"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G_DH" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.12 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.0
        done
    done
else
    echo "  [SKIP] G-DH (ENABLE_G_DH=0)"
fi

# ================================================================
# 세트 G-DHH: weight-diversity-loss=0.16, target-diversity-var=0.06
#
#   목적: div-w 강화의 포화점 확인.
#         G-DH(0.12)보다 더 강하게 분화를 강요했을 때 개선이 이어지는지,
#         혹은 과도한 압력으로 오히려 성능이 꺾이는지 탐색.
#   변경: --weight-diversity-loss 0.16
# ================================================================
if [ "${ENABLE_G_DHH}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G-DHH: div-w=0.16, div-v=0.06, g-clip=0.0"
    echo "  비교: G-DH (div-w=0.12) vs G-DHH → div-w 포화 여부"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G_DHH" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.16 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.0
        done
    done
else
    echo "  [SKIP] G-DHH (ENABLE_G_DHH=0)"
fi

# ================================================================
# 세트 G-DV: weight-diversity-loss=0.08, target-diversity-var=0.09
#
#   목적: div-v 단독 증가 효과 확인.
#         Ours(0.06)와 div-w/g-clip 완전 동일, div-v만 증가.
#         더 극단적인 레벨 집중(목표 분산 상향)이 유효한지 탐색.
#   변경: --target-diversity-var 0.09
# ================================================================
if [ "${ENABLE_G_DV}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G-DV: div-w=0.08, div-v=0.09, g-clip=0.0"
    echo "  비교: Ours (div-v=0.06) vs G-DV → div-v 단독 증가 효과"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G_DV" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var  0.09 \
                --gating-grad-clip      0.0
        done
    done
else
    echo "  [SKIP] G-DV (ENABLE_G_DV=0)"
fi

# ================================================================
# 세트 G-DHV: weight-diversity-loss=0.12, target-diversity-var=0.09
#
#   목적: div-w + div-v 교호작용 확인.
#         G-DHV > (G-DH 개선 + G-DV 개선)의 합 → 상호 시너지.
#         G-DHV ≈ G-DH 또는 G-DV 중 하나  → 한 축이 지배적.
#         G-DHV < Ours                     → 교호작용 부적(음의 시너지).
#   변경: --weight-diversity-loss 0.12 / --target-diversity-var 0.09
# ================================================================
if [ "${ENABLE_G_DHV}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G-DHV: div-w=0.12, div-v=0.09, g-clip=0.0"
    echo "  비교: G-DH + G-DV vs G-DHV → div 교호작용 방향"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G_DHV" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.12 \
                --target-diversity-var  0.09 \
                --gating-grad-clip      0.0
        done
    done
else
    echo "  [SKIP] G-DHV (ENABLE_G_DHV=0)"
fi

# ================================================================
# 세트 G-C2: gating-grad-clip=0.1
#
#   목적: grad-clip 강한 제약 구간 탐색.
#         clip=0.1이면 gating gradient norm이 0.1을 초과할 때마다
#         clip이 발동 → 비교적 자주 발동되는 강한 제약.
#         clip이 gating 수렴 안정성을 개선하는지 확인.
#   변경: --gating-grad-clip 0.1
# ================================================================
if [ "${ENABLE_G_C2}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G-C2: div-w=0.08, div-v=0.06, g-clip=0.1"
    echo "  비교: Ours (g-clip=0.0) vs G-C2 → grad-clip 강한 제약 효과"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G_C2" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.1
        done
    done
else
    echo "  [SKIP] G-C2 (ENABLE_G_C2=0)"
fi

# ================================================================
# 세트 G-C: gating-grad-clip=0.5
#
#   목적: grad-clip 약한 제약 구간 탐색.
#         clip=0.5이면 gradient norm이 매우 클 때만 발동.
#         G-C2(0.1) vs G-C(0.5) vs Ours(0.0) 세 점으로
#         clip 효과의 방향과 적정 범위를 한 번에 파악.
#   변경: --gating-grad-clip 0.5
# ================================================================
if [ "${ENABLE_G_C}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G-C: div-w=0.08, div-v=0.06, g-clip=0.5"
    echo "  비교: Ours (g-clip=0.0) vs G-C2 (0.1) vs G-C (0.5)"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G_C" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.5
        done
    done
else
    echo "  [SKIP] G-C (ENABLE_G_C=0)"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가 실행
# Ours 기준: recon_IC/ (기존 결과 재사용, 재실행 없음)
# 총 7세트 비교
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  평가 시작: evaluate_batch_v2.py"
    echo "  총 7세트: Ours / G_DH / G_DHH / G_DV / G_DHV / G_C2 / G_C"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    # 💡 이 부분의 python을 가상환경 절대경로로 변경했습니다.
    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${OURS_RECON_ROOT}:Ours" \
            "${RECON_ROOT}/G_DH:G_DH" \
            "${RECON_ROOT}/G_DHH:G_DHH" \
            "${RECON_ROOT}/G_DV:G_DV" \
            "${RECON_ROOT}/G_DHV:G_DHV" \
            "${RECON_ROOT}/G_C2:G_C2" \
            "${RECON_ROOT}/G_C:G_C" \
        --degraded-root "${DEGRADED_ROOT}" \
        --test-csv      "${TEST_CSV}" \
        --modalities    flair t1ce \
        --output-csv    eval_gating_sweep.csv

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_gating_sweep.csv"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 최종 요약 ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패한 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo "================================================================"

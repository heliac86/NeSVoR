#!/usr/bin/env bash
# ================================================================
# run_G9_full_run.sh  (v1 — 2026-05-27)
# P2_LR2 및 P7_LR2_DW10 전체 96-case full run
#
# ── 실험 배경 ────────────────────────────────────────────────────
# G8 파일럿(20-case) 결과:
#   P2_LR2    (lr-scale=2.0, div-w=0.08): FLAIR +0.113 dB / T1CE -0.006 dB
#   P6_DW10   (lr-scale=1.0, div-w=0.10): FLAIR +0.026 dB / T1CE +0.013 dB
#
# P7_LR2_DW10: P2(lr-scale 2.0) + P6(div-w 0.10) 복합 조합
#   → FLAIR 개선(lr-scale)과 T1CE 미약 개선(div-w) 두 축이
#     독립적으로 작용할 경우 양쪽 모달리티 동시 개선 기대.
#
# ── 비교 구성 ────────────────────────────────────────────────────
# G_C2      (기준, recon_gating_sweep/G_C2 재사용)
# P2_LR2    (신규 96-case)
# P7_LR2_DW10 (신규 96-case)
# → 3파전 비교: eval_G9_full_run.csv 출력
#
# ── 평균 지표 전략 ───────────────────────────────────────────────
# FLAIR + T1CE 평균 PSNR 기준 no_gating 대비 격차 주장 가능.
# no_gating 기준: FLAIR 26.643 / T1CE 26.332 → 평균 26.487 dB
# G_C2 기준:      FLAIR 26.763 / T1CE 26.373 → 평균 26.568 dB
#
# ── 세트 설계 ────────────────────────────────────────────────────
# 세트명          div-w  div-v  g-clip  lr-scale  목적
# G_C2(기준)      0.08   0.06   0.1     1.0       재사용
# P2_LR2          0.08   0.06   0.1     2.0       lr-scale 상향 효과
# P7_LR2_DW10     0.10   0.06   0.1     2.0       복합: lr-scale + div-w
#
# ── 총 실행 ─────────────────────────────────────────────────────
# 신규 2세트 × 96 cases = 192 케이스
# G_C2 포함 3세트 비교 (G_C2는 재실행 없음)
# 예상 총 시간: 신규 2세트 × ~7시간 ≈ 14시간
#
# ── 데이터 ──────────────────────────────────────────────────────
# 환자: test.csv 기반 48명 전체
# 모달리티: flair, t1ce
# 총 케이스: 48명 × 2모달 = 96 cases / 세트
#
# ── 사용법 ──────────────────────────────────────────────────────
#   chmod +x run_G9_full_run.sh
#   ./run_G9_full_run.sh                   # 학습 + 평가 전체
#   SKIP_TRAIN=1 ./run_G9_full_run.sh      # 평가만 재실행
#   SKIP_EVAL=1  ./run_G9_full_run.sh      # 학습만 실행
#   ENABLE_P2_LR2=0    ./run_G9_full_run.sh  # P2_LR2 건너뜀
#   ENABLE_P7_LR2_DW10=0 ./run_G9_full_run.sh  # P7 건너뜀
#
# ── 출력 파일 명명 규칙 ─────────────────────────────────────────
#   recon_G9_full/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────────
ENABLE_P2_LR2=${ENABLE_P2_LR2:-1}
ENABLE_P7_LR2_DW10=${ENABLE_P7_LR2_DW10:-1}

# ── 실행 단계 플래그 ────────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ───────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# G_C2 기준: run_gating_sweep.sh에서 생성된 결과 재사용
G_C2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_gating_sweep/G_C2"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── BASE_ARGS: G_C2 확정 설정 재현 ─────────────────────────────
# div-w, lr-scale만 세트별로 덮어씀
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
  --target-diversity-var 0.06
  --gating-grad-clip 0.1
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

# ── test.csv에서 환자 ID 로드 ──────────────────────────────────
mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

# ── 단일 케이스 실행 함수 ───────────────────────────────────────
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
# 세트 P2_LR2: gating-lr-scale=2.0, weight-diversity-loss=0.08
#
#   목적: G8 파일럿에서 FLAIR +0.113 dB를 기록한 설정의 재현성 확인.
#         lr-scale을 2배로 높여 gating이 L7 중심으로 빠르게 수렴.
#         48명 × 2 모달 = 96 케이스 전체에서 개선이 유지되는지 검증.
#   변경: --gating-lr-scale 2.0
#   비교: G_C2 (lr-scale=1.0) vs P2_LR2 → lr-scale 상향 효과 풀 검증
# ================================================================
if [ "${ENABLE_P2_LR2}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P2_LR2: div-w=0.08, div-v=0.06, g-clip=0.1, lr-scale=2.0"
    echo "  비교: G_C2 (lr-scale=1.0) vs P2_LR2 → lr-scale 상향 전체 검증"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P2_LR2" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --gating-lr-scale       2.0
        done
    done
else
    echo "  [SKIP] P2_LR2 (ENABLE_P2_LR2=0)"
fi

# ================================================================
# 세트 P7_LR2_DW10: gating-lr-scale=2.0, weight-diversity-loss=0.10
#
#   목적: P2(lr-scale 2.0)와 P6(div-w 0.10)의 복합 조합.
#         두 개선 축이 독립적으로 작용한다면:
#           - lr-scale 2.0 → FLAIR 개선 기여
#           - div-w 0.10   → T1CE 미약 개선 기여
#         → 양쪽 모달리티에서 동시 개선 기대.
#         G_C2, P2_LR2와의 3파전에서 복합 설정의 우위 여부 확인.
#   변경: --gating-lr-scale 2.0 / --weight-diversity-loss 0.10
#   비교: G_C2 vs P2_LR2 vs P7_LR2_DW10
#         → div-w 단독 증가(P6 파일럿)가 lr-scale 환경에서도 유효한가
# ================================================================
if [ "${ENABLE_P7_LR2_DW10}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P7_LR2_DW10: div-w=0.10, div-v=0.06, g-clip=0.1, lr-scale=2.0"
    echo "  비교: G_C2 / P2_LR2 / P7_LR2_DW10 → 복합 설정 효과"
    echo "  케이스: 96  예상 시간: ~7시간"
    echo "################################################################"
    for FULL_ID in "${FULL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P7_LR2_DW10" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.10 \
                --gating-lr-scale       2.0
        done
    done
else
    echo "  [SKIP] P7_LR2_DW10 (ENABLE_P7_LR2_DW10=0)"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가 실행
# G_C2 기준: recon_gating_sweep/G_C2/ (기존 결과 재사용)
# 총 3세트 비교
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  평가 시작: evaluate_batch_v2.py"
    echo "  총 3세트: G_C2 / P2_LR2 / P7_LR2_DW10"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${G_C2_RECON_ROOT}:G_C2"                         \
            "${RECON_ROOT}/P2_LR2:P2_LR2"                     \
            "${RECON_ROOT}/P7_LR2_DW10:P7_LR2_DW10"           \
        --degraded-root "${DEGRADED_ROOT}"                     \
        --test-csv      "${TEST_CSV}"                          \
        --modalities    flair t1ce                             \
        --output-csv    eval_G9_full_run.csv

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_G9_full_run.csv"
        echo ""
        echo "  ── 비교 기준점 (참고) ───────────────────────────────────"
        echo "  no_gating:  FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB"
        echo "  G_C2:       FLAIR 26.763 / T1CE 26.373 / 평균 26.568 dB"
        echo "  G_C2 pilot: FLAIR 27.143 / T1CE 26.409 (20-case 기준)"
        echo "  ─────────────────────────────────────────────────────────"
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
echo ""
echo "  ── 실험 메모 ─────────────────────────────────────────────"
echo "  기준 브랜치 : backup_IC"
echo "  G_C2 경로   : ${G_C2_RECON_ROOT}"
echo "  결과 경로   : ${RECON_ROOT}"
echo "  결과 CSV    : eval_G9_full_run.csv"
echo "================================================================"

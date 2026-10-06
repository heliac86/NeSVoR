#!/usr/bin/env bash
# ================================================================
# run_G12_gpu2.sh  (v1 — 2026-05-28)  [GPU 2번 전용]
# 세트: G12_h4 — spatial gating, hidden_dim=4, no-z-gating (G6 모드)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G11_h8 (파일럿 10명, G6 모드):
#   FLAIR 27.407 / T1CE 26.458 / 평균 26.932 dB  ← P2_LR2 양쪽 모두 초과
# P2_LR2 (파일럿 10명):
#   FLAIR 27.255 / T1CE 26.403 / 평균 26.829 dB
#
# G11에서 확인된 방향성:
#   hidden_dim이 줄어들수록 결과가 좋아지는 일관된 경향 (h32→h16→h8)
#   → 2000 iter 스케줄에서 MLP가 클수록 수렴 부족이라는 구조적 신호
#   → h4가 h8보다 나으면: 더 단순화가 답
#   → h4 ≈ h8이면: h8이 수렴 포화점
#
# ── 이번 실험 목적 ────────────────────────────────────────────
# Q1. h4 (≈76 파라미터) 가 h8 (≈136 파라미터) 보다 나은가?
#     → hidden_dim 방향성 최종 확인
# Q2. 96-case에서도 파일럿(10명)과 동일한 방향성이 유지되는가?
#     → G11_h8 파일럿 결과의 재현성 검증
#
# ── G6 모드 고정 이유 ─────────────────────────────────────────
# GPU3(run_G12_gpu3.sh)에서 h8 + z-gating(G7 모드)을 병렬 실험 중.
# 이 파일은 G6(좌표만) vs G7(좌표+z) 비교의 G6 쪽 대조군 역할도 겸함.
# h4는 Linear(3→4)→ReLU→Linear(4→n_levels)로
# GatingMLP 파라미터 수가 극단적으로 적어 단기 스케줄에 최적화됨.
#
# ── 스케일 ────────────────────────────────────────────────────
# 전체 96 cases (파일럿 아님). 7시간 여유 내 완료 목표.
# 예상 시간: 96 cases × 2 modalities × ~4.5분/case ≈ ~14시간
# → 사전에 GPU 점유 상태 확인 필요. 필요 시 세트 수 조정.
# ※ 실제 소요 시간은 환경에 따라 다를 수 있음
#
# ── 비교 기준점 (96-case 전체) ────────────────────────────────
#   no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB
#   P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB
#
# ── 판단 기준 ─────────────────────────────────────────────────
#   FLAIR AND T1CE 양쪽 P2_LR2 이상 → full run 완료, 논문 결과 확정
#   FLAIR만 개선, T1CE 동등         → G12_h8z 결과와 종합 판단
#   양쪽 모두 P2_LR2 미만           → h4 방향 종료, 활성화 함수 탐색
#
# ── 사용법 ───────────────────────────────────────────────────
#   chmod +x run_G12_gpu2.sh
#   CUDA_VISIBLE_DEVICES=2 ./run_G12_gpu2.sh
#   SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=2 ./run_G12_gpu2.sh  # 평가만
#   SKIP_EVAL=1  CUDA_VISIBLE_DEVICES=2 ./run_G12_gpu2.sh  # 학습만
#
# ── 출력 파일 명명 규칙 ───────────────────────────────────────
#   recon_G12/G12_h4/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_G12_h4.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
EXP_TAG="G12_h4"
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G12"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# P2_LR2 기준: recon_G9_full/P2_LR2 재사용 (이미 96-case 완료)
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── 전체 환자 ID 로드 (96 cases) ─────────────────────────────
mapfile -t ALL_IDS < <(
    tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r' | grep -v '^$'
)

# ── 파일럿 환자 10명 (seed=42, 평가 분리용) ──────────────────
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
echo "  전체 케이스: ${#ALL_IDS[@]}명 × 2 modalities = $((${#ALL_IDS[@]} * 2)) cases"
echo "  파일럿 (평가 분리용, seed=42): ${#PILOT_IDS[@]}명"
echo "  실험 태그: ${EXP_TAG}"
echo "================================================================"

# ── 공통 인자: P2_LR2 설정 + spatial-gating(G6) ──────────────
BASE_ARGS="
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
  --gating-lr-scale 2.0
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
  --spatial-gating
  --no-z-gating
  --gating-hidden-dim 4
"
# --spatial-gating + --no-z-gating → G6 모드 (좌표만, single-pass)
# --gating-hidden-dim 4: Linear(3→4)→ReLU→Linear(4→n_levels) ≈ 76 파라미터

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ────────────────────────────────────
run_case() {
    local FULL_ID=$1
    local MODALITY=$2

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
    echo "  [START] ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${BASE_ARGS}

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}  $(date '+%H:%M:%S')"
    fi
}

# ================================================================
# 학습: 전체 96 cases
# ================================================================
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  학습 시작: ${EXP_TAG}  (96 cases × 2 modalities)"
    echo "  hidden_dim=4, n-iter=2000, G6 모드(no-z-gating)"
    echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"
    for FULL_ID in "${ALL_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "${FULL_ID}" "${MODALITY}"
        done
    done
    echo ""
    echo "  학습 완료: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  성공: ${SUCCESS} / ${TOTAL}"
fi

# ================================================================
# 평가: 파일럿(10명) + 전체(96 cases) 분리 실행
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    # ── 파일럿 CSV 생성 ──────────────────────────────────────
    PILOT_CSV="/tmp/pilot_test_G12.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    # ── 평가 1: 파일럿 10명 (빠른 중간 확인) ────────────────
    echo ""
    echo "################################################################"
    echo "  평가 1/2: 파일럿 10명 (중간 확인용)"
    echo "  비교: P2_LR2 / G12_h4 / G12_h8z(GPU3)"
    echo "  ※ G12_h8z는 GPU3 완료 후 경로 접근 가능"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                          \
            "${RECON_ROOT}/G12_h4:G12_h4"                           \
            "${RECON_ROOT}/G12_h8z:G12_h8z"                         \
        --degraded-root "${DEGRADED_ROOT}"                           \
        --test-csv      "${PILOT_CSV}"                               \
        --modalities    flair t1ce                                   \
        --output-csv    eval_G12_pilot.csv

    if [ $? -eq 0 ]; then
        echo "  ✅ 파일럿 평가 완료 → eval_G12_pilot.csv"
    else
        echo "  ⚠ 파일럿 평가 실패 (G12_h8z 미완료일 수 있음, 계속 진행)"
    fi

    # ── 평가 2: 96-case 전체 (최종 판단 기준) ───────────────
    echo ""
    echo "################################################################"
    echo "  평가 2/2: 96-case 전체 (최종 판단)"
    echo "  비교: P2_LR2 / G12_h4 / G12_h8z(GPU3)"
    echo "  ※ GPU3 완료 후 실행 권장. G12_h8z 미완료 시 해당 세트 스킵됨."
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                          \
            "${RECON_ROOT}/G12_h4:G12_h4"                           \
            "${RECON_ROOT}/G12_h8z:G12_h8z"                         \
        --degraded-root "${DEGRADED_ROOT}"                           \
        --test-csv      "${TEST_CSV}"                                \
        --modalities    flair t1ce                                   \
        --output-csv    eval_G12_full96.csv

    if [ $? -eq 0 ]; then
        echo "  ✅ 96-case 평가 완료 → eval_G12_full96.csv"
        echo ""
        echo "  ── 비교 기준점 (96-case 전체) ──────────────────────────"
        echo "  no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB"
        echo "  P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB"
        echo "  ── G11 파일럿(10명) 참고 ───────────────────────────────"
        echo "  G11_h8(G6): FLAIR 27.407 / T1CE 26.458 / 평균 26.932 dB"
        echo "  P2_LR2:     FLAIR 27.255 / T1CE 26.403 / 평균 26.829 dB"
        echo "  ─────────────────────────────────────────────────────────"
        echo "  ── 판단 기준 ────────────────────────────────────────────"
        echo "  G12_h4 양쪽 P2_LR2 이상 → full run 완료, 논문 결과 확정"
        echo "  FLAIR만 개선             → GPU3 결과와 종합 판단"
        echo "  양쪽 P2_LR2 미만         → SiLU 활성화 함수 탐색 진행"
        echo "  ─────────────────────────────────────────────────────────"
    else
        echo "  ❌ 96-case 평가 실패"
        echo "  GPU3 완료 후 재실행: SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=2 ./run_G12_gpu2.sh"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 최종 요약 ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  [GPU2] 전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
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
echo "  실험 경로     : ${RECON_ROOT}/${EXP_TAG}"
echo "  파일럿 결과   : eval_G12_pilot.csv  (10명, 중간 확인)"
echo "  전체 결과     : eval_G12_full96.csv (96 cases, 최종 판단)"
echo "  G12_h4        : hidden_dim=4, n-iter=2000, G6(no-z-gating)"
echo "  G12_h8z(GPU3) : hidden_dim=8, n-iter=2000, G7(z-gating ON)"
echo "================================================================"

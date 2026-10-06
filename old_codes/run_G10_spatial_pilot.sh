#!/usr/bin/env bash
# ================================================================
# run_G10_spatial_pilot.sh  (v1 — 2026-05-28)
# Spatial GatingMLP 재시도 파일럿 (P2_LR2 직접 비교)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G6/G7 당시 spatial gating이 전역 gating(G5_lowvar)보다 낮게
# 나왔던 원인을 사후 분석한 결과, 다음 세 가지 조건 미달이
# 복합 작용한 것으로 판단:
#
#   1. gating-grad-clip=0.0  → L7 gradient 폭발 억제 없음
#   2. gating-lr-scale=1.0   → 2000 iter 내 GatingMLP 수렴 부족
#   3. weight-ff-loss=0.10 / ff-alpha=1.0  → 현재 최적 설정과 상이
#
# 현재 G1 최적 설정(P2_LR2: clip=0.1, lr-scale=2.0, ff=0.05/2.0)을
# 그대로 유지한 채 --spatial-gating 플래그만 추가하여,
# 조건 보정 후 spatial gating의 재현 가능성을 검증.
#
# warmup은 일절 사용하지 않음:
#   G6/G7에서 warmup=500이 오히려 FLAIR -2.7 dB 손실을 초래.
#   2000 iter 중 25%를 미학습 hash grid 출력으로 GatingMLP를
#   훈련하는 것은 이 스케줄에서 역효과임이 확인됨.
#
# ── 탐색 설계 ──────────────────────────────────────────────────
# 기준: P2_LR2 (G1 전역 gating, recon_G9_full/P2_LR2 재사용)
#
# 세트명       hidden_dim  warmup  탐색 목적
# ---------    ----------  ------  --------
# P2_LR2(기준) G1 전역     —       재사용 (재실행 없음)
# G10_h16      16          0       최소 용량 — 2000 iter 수렴 안정성 우선
# G10_h32      32          0       G6 당시 탐색값 — 조건 보정 후 재시도
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q1. G1 최적 조건(clip=0.1, lr=2.0)에서 spatial gating이
#     전역 gating(P2_LR2)을 넘는가?
#     → G10_h16 / G10_h32  vs  P2_LR2
#
# Q2. hidden_dim=16(단순)과 32(기존 탐색값) 중 어느 쪽이
#     2000 iter 단기 스케줄에 더 적합한가?
#     → G10_h16 vs G10_h32
#
# ── diversity loss 처리 ────────────────────────────────────────
# train.py: spatial_gating=True 시 전역 diversity loss 자동 비활성화
#   (use_diversity_loss = ... and not spatial_gating)
# → --weight-diversity-loss 인자 불필요. 명시 생략.
#
# ── 판단 기준 ──────────────────────────────────────────────────
# FLAIR AND T1CE 양쪽 모두 P2_LR2 이상 → 96-case full run 격상
# FLAIR만 개선, T1CE는 동등 이상         → 후보 유지, 추가 탐색 검토
# 양쪽 모두 P2_LR2 미만                  → spatial gating 방향 종료
#
# ── 환자 선정 ──────────────────────────────────────────────────
# G8 파일럿과 동일한 seed=42 기반 10명 (재현성 보장)
# 모달리티: flair, t1ce → 총 20 cases / 세트
#
# ── 총 실행 ─────────────────────────────────────────────────
# 신규 2세트 × 20 cases = 40 케이스
# P2_LR2 기준 포함 3세트 비교
# 예상 총 시간: 신규 2세트 × ~1.5시간 ≈ 3시간
#
# ── 사용법 ──────────────────────────────────────────────────
#   chmod +x run_G10_spatial_pilot.sh
#   ./run_G10_spatial_pilot.sh                  # 학습 + 평가 전체
#   SKIP_TRAIN=1 ./run_G10_spatial_pilot.sh     # 평가만 재실행
#   SKIP_EVAL=1  ./run_G10_spatial_pilot.sh     # 학습만 실행
#   ENABLE_G10_H16=0 ./run_G10_spatial_pilot.sh # G10_h16 건너뜀
#   ENABLE_G10_H32=0 ./run_G10_spatial_pilot.sh # G10_h32 건너뜀
#
# ── 출력 파일 명명 규칙 ──────────────────────────────────────
#   recon_G10_spatial/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_G10_H16=${ENABLE_G10_H16:-1}
ENABLE_G10_H32=${ENABLE_G10_H32:-1}

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G10_spatial"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# P2_LR2 기준: recon_G9_full/P2_LR2 재사용
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── seed=42 기반 파일럿 환자 10명 (G8 파일럿과 동일) ─────────
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
echo "  파일럿 환자 목록 (seed=42, 10명 — G8과 동일):"
for ID in "${PILOT_IDS[@]}"; do
    echo "    - ${ID}"
done
echo "================================================================"

# ── BASE_ARGS: P2_LR2 설정 완전 재현 + --spatial-gating 추가 ──
# G1(P2_LR2) 대비 변경점: --spatial-gating 플래그만 추가.
# gating-grad-clip, gating-lr-scale, ff 설정은 P2_LR2와 동일.
# weight-diversity-loss는 train.py에서 spatial_gating=True 시
# 자동 비활성화되므로 명시하지 않음.
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
"
# 주: diversity 관련 인자들은 spatial_gating=True 시 train.py 내부에서
#     use_diversity_loss=False로 처리되므로 실질적으로 무효화됨.
#     P2_LR2와의 BASE_ARGS 일관성을 위해 명시적으로 포함.

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
# 세트 G10_h16: spatial-gating + hidden_dim=16
#
#   목적: 최소 용량 GatingMLP로 2000 iter 내 안정적 수렴 확인.
#         G6 당시 h32/h64 모두 전역 gating보다 낮았던 원인 중 하나로
#         단기 스케줄에서의 underfitting(수렴 부족)을 의심.
#         더 단순한 모델(hidden_dim=16)이 짧은 스케줄에 더 잘 맞는지 검증.
#         GatingMLP params: Linear(3→16) + ReLU + Linear(16→n_levels)
#                           = 3*16+16 + 16*n_levels+n_levels ≈ 256개 (h32의 절반)
#   변경: --gating-hidden-dim 16
#   비교: P2_LR2 (전역 gating) vs G10_h16 → spatial gating 최소 버전
# ================================================================
if [ "${ENABLE_G10_H16}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G10_h16: spatial-gating, hidden_dim=16, warmup=0"
    echo "  gating-grad-clip=0.1 / gating-lr-scale=2.0 (P2_LR2 동일)"
    echo "  비교: P2_LR2 (전역) vs G10_h16 → spatial gating 효과"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G10_h16" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --gating-hidden-dim 16
        done
    done
else
    echo "  [SKIP] G10_h16 (ENABLE_G10_H16=0)"
fi

# ================================================================
# 세트 G10_h32: spatial-gating + hidden_dim=32
#
#   목적: G6에서 탐색했던 hidden_dim=32를 현재 최적 G1 조건
#         (clip=0.1, lr-scale=2.0, ff=0.05/2.0)으로 재시도.
#         당시 실패 원인이 조건 미달(clip 부재, lr-scale 부족)이었다면
#         이번엔 전역 gating을 넘는 결과가 나올 것으로 기대.
#         GatingMLP params: Linear(3→32) + ReLU + Linear(32→n_levels)
#                           ≈ 500개
#   변경: --gating-hidden-dim 32
#   비교: P2_LR2 (전역) vs G10_h32 → 조건 보정 후 재검증
#         G10_h16 vs G10_h32        → hidden_dim 방향성 파악
# ================================================================
if [ "${ENABLE_G10_H32}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 G10_h32: spatial-gating, hidden_dim=32, warmup=0"
    echo "  gating-grad-clip=0.1 / gating-lr-scale=2.0 (P2_LR2 동일)"
    echo "  비교: P2_LR2 (전역) vs G10_h32 → G6 조건 보정 재시도"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G10_h32" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --gating-hidden-dim 32
        done
    done
else
    echo "  [SKIP] G10_h32 (ENABLE_G10_H32=0)"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가 실행
# P2_LR2 기준: recon_G9_full/P2_LR2 (기존 결과 재사용, 재실행 없음)
# 단, P2_LR2의 경우 파일럿 10명에 해당하는 케이스만 필터링하여 비교
# → G8 파일럿과 동일한 방식 (임시 파일럿 CSV 생성)
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    # 파일럿 환자 ID를 임시 CSV로 저장 (헤더 포함)
    PILOT_CSV="/tmp/pilot_test_G10.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    echo ""
    echo "################################################################"
    echo "  평가 시작: evaluate_batch_v2.py"
    echo "  총 3세트: P2_LR2(기준) / G10_h16 / G10_h32"
    echo "  파일럿 CSV: ${PILOT_CSV} (10명 × 2 모달 = 20 cases)"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"            \
            "${RECON_ROOT}/G10_h16:G10_h16"           \
            "${RECON_ROOT}/G10_h32:G10_h32"           \
        --degraded-root "${DEGRADED_ROOT}"             \
        --test-csv      "${PILOT_CSV}"                 \
        --modalities    flair t1ce                     \
        --output-csv    eval_G10_spatial_pilot.csv

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_G10_spatial_pilot.csv"
        echo ""
        echo "  ── 다음 단계 판단 기준 ──────────────────────────────"
        echo "  P2_LR2 기준 (파일럿 10명, FLAIR/T1CE 양쪽 확인)"
        echo "  FLAIR AND T1CE 양쪽 P2_LR2 이상 → 96-case full run 격상"
        echo "  FLAIR만 개선, T1CE 동등 이상     → 후보 유지, 추가 탐색"
        echo "  양쪽 모두 P2_LR2 미만            → spatial gating 방향 종료"
        echo "  ────────────────────────────────────────────────────"
        echo ""
        echo "  ── 비교 기준점 (참고, 96-case 전체 기준) ───────────"
        echo "  no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB"
        echo "  P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB"
        echo "  ────────────────────────────────────────────────────"
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
echo "  ── 실험 메모 ──────────────────────────────────────────────"
echo "  기준 브랜치 : backup_IC"
echo "  P2_LR2 경로 : ${P2_LR2_RECON_ROOT}"
echo "  실험 경로   : ${RECON_ROOT}"
echo "  결과 CSV    : eval_G10_spatial_pilot.csv"
echo "  파일럿 CSV  : /tmp/pilot_test_G10.csv (임시 파일, 재실행 시 자동 재생성)"
echo "================================================================"

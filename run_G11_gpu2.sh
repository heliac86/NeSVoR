#!/usr/bin/env bash
# ================================================================
# run_G11_gpu2.sh  (v1 — 2026-05-28)  [GPU 2번 전용]
# 세트 A: G11_h8       — spatial gating, hidden_dim=8,  n-iter=2000
# 세트 B: G11_h16_2500 — spatial gating, hidden_dim=16, n-iter=2500
# 평가: P2_LR2(기준) / G11_h8 / G11_h16_2500 3파전
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G10 결과 (파일럿 10명):
#   G10_h16: FLAIR 27.123 / T1CE 26.599 / 평균 26.861 dB
#   P2_LR2:  FLAIR 27.255 / T1CE 26.403 / 평균 26.829 dB
#   → h16이 T1CE에서 +0.196 dB 우위, FLAIR에서 -0.132 dB 열위
#
# ⚠ G10 설계 오류 발견:
#   --no-z-gating 미지정으로 G10_h16/h32가 G6 모드가 아닌
#   G7 2-pass 모드(좌표+z_repr)로 실행됨.
#   G7이 G6보다 나빴던 이력 있음 → 이번 G11부터 --no-z-gating 명시.
#
# ── 세트 A: G11_h8 (hidden_dim=8, n-iter=2000) ─────────────────
#   목적: h16보다 더 단순한 MLP가 2000 iter에 더 적합한지 확인.
#         h8 > h16 → 2000 iter에서 h16도 underfitting 중
#         h8 < h16 → h16이 표현력 하한에 근접
#   변경: --gating-hidden-dim 8
#   비교: G10_h16(G7모드) vs G11_h8(G6모드) — 모드 차이 주의
#         주요 비교는 P2_LR2 기준으로 판단
#
# ── 세트 B: G11_h16_2500 (hidden_dim=16, n-iter=2500) ──────────
#   목적: h16에서 iter를 2000→2500으로 25% 늘렸을 때
#         FLAIR 손실이 회복되는지 확인.
#         G6/G7은 n-iter=6000이었고 warmup=500에서도 수렴 문제.
#         → GatingMLP가 전역 level_weights보다 수렴이 느린 구조임을
#           감안하면 iter 소폭 증가가 가장 직접적인 해결책.
#   변경: --n-iter 2500
#   비교: G10_h16(n-iter=2000) vs G11_h16_2500 → iter 증가 효과
#         G11_h8 vs G11_h16_2500 → hidden_dim vs iter 교호 효과
#
# ── no-z-gating 명시 이유 ──────────────────────────────────────
#   G6 모드(좌표만): Linear(3→hidden) → ReLU → Linear(hidden→n_levels)
#   G7 모드(좌표+z): 2-pass forward → z_repr detach → cat → GatingMLP
#   G10에서 z-gating이 의도치 않게 ON된 것을 이번에 수정.
#   G6 vs G7 순수 비교가 목적이 아니므로 G6 모드(단순, 안정)로 고정.
#
# ── GPU 배정 ──────────────────────────────────────────────────
#   이 파일: GPU 2번
#   CUDA_VISIBLE_DEVICES=2 ./run_G11_gpu2.sh
#
# ── 판단 기준 ─────────────────────────────────────────────────
#   FLAIR AND T1CE 양쪽 P2_LR2 이상 → 96-case full run 격상
#   FLAIR만 개선, T1CE 동등         → 후보 유지, 추가 탐색
#   양쪽 모두 P2_LR2 미만           → 해당 방향 종료
#
# ── 비교 기준점 (참고, 96-case 전체 기준) ─────────────────────
#   no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB
#   P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB
#
# ── 총 실행 ──────────────────────────────────────────────────
#   세트 A: 20 cases (~1.5시간)
#   세트 B: 20 cases (~1.9시간, n-iter 25% 증가)
#   합계: ~3.4시간 + 평가
#
# ── 사용법 ───────────────────────────────────────────────────
#   chmod +x run_G11_gpu2.sh
#   CUDA_VISIBLE_DEVICES=2 ./run_G11_gpu2.sh
#   SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=2 ./run_G11_gpu2.sh  # 평가만
#   ENABLE_G11_H8=0      CUDA_VISIBLE_DEVICES=2 ./run_G11_gpu2.sh
#   ENABLE_G11_H16_2500=0 CUDA_VISIBLE_DEVICES=2 ./run_G11_gpu2.sh
#
# ── 출력 파일 명명 규칙 ──────────────────────────────────────
#   recon_G11/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ──────────────────────────────────────
ENABLE_G11_H8=${ENABLE_G11_H8:-1}
ENABLE_G11_H16_2500=${ENABLE_G11_H16_2500:-1}

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G11"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# P2_LR2 기준: recon_G9_full/P2_LR2 재사용
P2_LR2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G9_full/P2_LR2"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── seed=42 기반 파일럿 환자 10명 (G8/G10과 동일) ───────────
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
echo "  파일럿 환자 목록 (seed=42, 10명 — G8/G10과 동일):"
for ID in "${PILOT_IDS[@]}"; do
    echo "    - ${ID}"
done
echo "================================================================"

# ── BASE_ARGS: P2_LR2 + --spatial-gating + --no-z-gating ─────
# G10 오류 수정: --no-z-gating 명시 → G6 순수 좌표 모드(single-pass)
# G11 공통 고정값. n-iter와 hidden-dim은 세트별로 덮어씀.
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
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
  --spatial-gating
  --no-z-gating
"
# 주: --spatial-gating + --no-z-gating → G6 모드 (좌표만, single-pass)
#     diversity 관련 인자는 train.py에서 spatial_gating=True시 자동 무효화됨

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
# 세트 A: G11_h8  [n-iter=2000, hidden_dim=8]
#
#   목적: h16보다 더 단순화된 MLP로 2000 iter 수렴 가능성 확인.
#         Linear(3→8) → ReLU → Linear(8→n_levels) ≈ 136개 파라미터
#         h16(≈256개)의 절반. 만약 h8이 h16보다 우수하다면
#         2000 iter에서 h16조차 수렴이 부족하다는 강한 증거.
#   변경: --gating-hidden-dim 8  /  --n-iter 2000
#   비교: P2_LR2 vs G11_h8 → spatial gating 최소 유효 용량 탐색
# ================================================================
if [ "${ENABLE_G11_H8}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 A: G11_h8 — hidden_dim=8, n-iter=2000, no-z-gating"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G11_h8" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --gating-hidden-dim 8 \
                --n-iter 2000
        done
    done
else
    echo "  [SKIP] G11_h8 (ENABLE_G11_H8=0)"
fi

# ================================================================
# 세트 B: G11_h16_2500  [n-iter=2500, hidden_dim=16]
#
#   목적: h16에서 iter를 25% 늘려 FLAIR 손실 회복 여부 확인.
#         G10_h16(G7 모드, n-iter=2000)이 FLAIR에서 -0.132 dB인 것이
#         (a) G7→G6 모드 전환만으로 해결되는지, 아니면
#         (b) iter 부족이 복합된 문제인지 분리 판단 가능.
#         GPU3의 P2_LR2_2500과 비교하면 공정한 iter 조건 비교 가능.
#   변경: --gating-hidden-dim 16  /  --n-iter 2500
#   비교: G10_h16(2000) vs G11_h16_2500 → iter 증가 효과
#         P2_LR2_2500(GPU3) vs G11_h16_2500 → 동일 iter에서 spatial 유효성
# ================================================================
if [ "${ENABLE_G11_H16_2500}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 B: G11_h16_2500 — hidden_dim=16, n-iter=2500, no-z-gating"
    echo "  케이스: 20  예상 시간: ~1.9시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G11_h16_2500" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --gating-hidden-dim 16 \
                --n-iter 2500
        done
    done
else
    echo "  [SKIP] G11_h16_2500 (ENABLE_G11_H16_2500=0)"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가 실행
# P2_LR2: recon_G9_full/P2_LR2 재사용 (파일럿 10명 필터링)
# G11_h8, G11_h16_2500: 이번 실험 결과
# GPU3의 P2_LR2_2500은 별도 평가 (run_G11_gpu3.sh에서 처리)
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    PILOT_CSV="/tmp/pilot_test_G11.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    echo ""
    echo "################################################################"
    echo "  평가 시작: evaluate_batch_v2.py  [GPU2 세트]"
    echo "  총 3세트: P2_LR2(기준) / G11_h8 / G11_h16_2500"
    echo "  파일럿 CSV: ${PILOT_CSV} (10명 × 2 모달 = 20 cases)"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                  \
            "${RECON_ROOT}/G11_h8:G11_h8"                   \
            "${RECON_ROOT}/G11_h16_2500:G11_h16_2500"       \
        --degraded-root "${DEGRADED_ROOT}"                   \
        --test-csv      "${PILOT_CSV}"                       \
        --modalities    flair t1ce                           \
        --output-csv    eval_G11_gpu2.csv

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_G11_gpu2.csv"
        echo ""
        echo "  ── 참고: G10 결과 (파일럿 10명, G7 모드) ──────────────"
        echo "  G10_h16: FLAIR 27.124 / T1CE 26.599 / 평균 26.861 dB"
        echo "  G10_h32: FLAIR 26.549 / T1CE 26.131 / 평균 26.340 dB"
        echo "  P2_LR2:  FLAIR 27.255 / T1CE 26.403 / 평균 26.829 dB"
        echo "  ────────────────────────────────────────────────────────"
        echo "  ── 비교 기준점 (96-case 전체 기준) ──────────────────────"
        echo "  no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB"
        echo "  P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB"
        echo "  ────────────────────────────────────────────────────────"
        echo "  ── GPU3 결과 (P2_LR2_2500)는 eval_G11_gpu3.csv 참조 ───"
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
echo "  기준 브랜치 : backup_IC"
echo "  P2_LR2 경로 : ${P2_LR2_RECON_ROOT}"
echo "  실험 경로   : ${RECON_ROOT}"
echo "  결과 CSV    : eval_G11_gpu2.csv"
echo "  G11_h8      : hidden_dim=8,  n-iter=2000, G6 모드(no-z-gating)"
echo "  G11_h16_2500: hidden_dim=16, n-iter=2500, G6 모드(no-z-gating)"
echo "================================================================"

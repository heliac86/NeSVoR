#!/usr/bin/env bash
# ================================================================
# run_G12_gpu3.sh  (v1 — 2026-05-28)  [GPU 3번 전용]
# 세트: G12_h8z — spatial gating, hidden_dim=8, z-gating ON (G7 모드)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# G11_h8 (파일럿 10명, G6 모드, no-z-gating):
#   FLAIR 27.407 / T1CE 26.458 / 평균 26.932 dB  ← P2_LR2 양쪽 초과
# G10_h16 (파일럿 10명, G7 모드, z-gating ON):
#   FLAIR 27.124 / T1CE 26.599 / 평균 26.861 dB
#
# 추론:
#   G6(좌표만)는 FLAIR에서 강하고 (G11_h8: +0.152 vs P2_LR2)
#   G7(좌표+z_repr)는 T1CE에서 강한 경향 (G10_h16: +0.196 vs P2_LR2)
#   → hidden_dim=8에서 z-gating을 켜면 두 장점이 합쳐질 가능성.
#   단, G10_h16은 hidden_dim=16이었으므로 h8+z의 결과는 미지수.
#
# ── 이번 실험 목적 ────────────────────────────────────────────
# Q1. h8 + z-gating(G7)이 h8(G6)보다 T1CE를 개선하는가?
#     → z_repr의 실제 기여 분리 검증
# Q2. FLAIR 강점을 유지하면서 T1CE도 개선되는가?
#     → 양쪽 모달리티 동시 개선 가능성 확인
# Q3. GPU2(G12_h4, G6)와 비교해 G6 vs G7 구조 차이 평가.
#     (h4 ≠ h8이므로 완전한 controlled 비교는 아님)
#
# ── G7 모드 동작 방식 (참고) ─────────────────────────────────
# Pass1: encoding → density_net → z_feat (gating 없이)
# z_repr = z_feat.mean(n_samples).detach()  ← gradient 차단
# gating_input = cat([x_repr(3), z_repr(n_feat_z)], dim=-1)
# GatingMLP(gating_input) → level logits → softmax*n_levels → weights
# Pass2: encoding → gated pe → density_net → 최종 출력
# → GatingMLP 입력 차원: 3 + n_features_z
# → hidden_dim=8: Linear(3+n_feat_z → 8) → ReLU → Linear(8 → n_levels)
#
# ── G10의 의도치 않은 G7 모드 결과 해석 ──────────────────────
# G10_h16이 G11_h8(G6)보다 T1CE에서 높은 것은
# hidden_dim 차이(16 vs 8)가 아닌 z-gating 효과일 가능성.
# 이번에 hidden_dim=8로 고정하고 z-gating만 ON해서 순수 효과 확인.
#
# ── GPU 배정 ──────────────────────────────────────────────────
#   이 파일: GPU 3번
#   GPU2(run_G12_gpu2.sh): G12_h4 (G6, no-z-gating)
#   이 파일: G12_h8z (G7, z-gating ON)
#   평가: GPU2에서 두 세트 합산 평가 (eval_G12_pilot.csv, eval_G12_full96.csv)
#
# ── 스케일 ────────────────────────────────────────────────────
# 전체 96 cases (파일럿 아님). 7시간 여유 내 완료 목표.
# G7 2-pass로 인해 G6 대비 ~20% 추가 연산 예상.
#
# ── 비교 기준점 (96-case 전체) ────────────────────────────────
#   no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB
#   P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB
#
# ── 사용법 ───────────────────────────────────────────────────
#   chmod +x run_G12_gpu3.sh
#   CUDA_VISIBLE_DEVICES=3 ./run_G12_gpu3.sh
#   SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=3 ./run_G12_gpu3.sh  # 평가만
#   SKIP_EVAL=1  CUDA_VISIBLE_DEVICES=3 ./run_G12_gpu3.sh  # 학습만
#
# ── 출력 파일 명명 규칙 ───────────────────────────────────────
#   recon_G12/G12_h8z/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_G12_h8z.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
EXP_TAG="G12_h8z"
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
echo "  실험 태그: ${EXP_TAG}  (G7 모드: 좌표 + z_repr)"
echo "================================================================"

# ── 공통 인자: P2_LR2 설정 + spatial-gating(G7, z-gating ON) ─
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
  --gating-hidden-dim 8
"
# --spatial-gating (--no-z-gating 없음) → G7 모드 (좌표 + z_repr, 2-pass)
# GatingMLP 입력: cat([x_repr(3), z_repr(n_feat_z)]) → hidden_dim=8
# z_repr은 density_net Pass1 출력에서 detach하여 가져옴

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
    echo "  hidden_dim=8, n-iter=2000, G7 모드(z-gating ON, 2-pass)"
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
# 평가: GPU3 자체 결과 단독 확인
# 파일럿(10명) + 전체(96 cases) 분리 실행
# 두 GPU 합산 평가는 run_G12_gpu2.sh의 SKIP_TRAIN=1로 재실행
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    # ── 파일럿 CSV 생성 ──────────────────────────────────────
    PILOT_CSV="/tmp/pilot_test_G12.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    # ── 평가 1: 파일럿 10명 (G12_h8z 단독) ──────────────────
    echo ""
    echo "################################################################"
    echo "  평가 1/2: 파일럿 10명 — G12_h8z 단독"
    echo "  비교: P2_LR2 / G11_h8(G6, 파일럿) / G12_h8z(G7)"
    echo "  ※ G11_h8 결과는 recon_G11 경로에서 참조"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    G11_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G11"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                          \
            "${G11_RECON_ROOT}/G11_h8:G11_h8_G6"                   \
            "${RECON_ROOT}/G12_h8z:G12_h8z_G7"                     \
        --degraded-root "${DEGRADED_ROOT}"                           \
        --test-csv      "${PILOT_CSV}"                               \
        --modalities    flair t1ce                                   \
        --output-csv    eval_G12_h8z_pilot.csv

    if [ $? -eq 0 ]; then
        echo "  ✅ 파일럿 평가 완료 → eval_G12_h8z_pilot.csv"
        echo "  → G6(G11_h8) vs G7(G12_h8z): z-gating 효과 확인 가능"
    else
        echo "  ⚠ 파일럿 평가 실패"
    fi

    # ── 평가 2: 96-case 전체 (G12_h8z 단독) ─────────────────
    echo ""
    echo "################################################################"
    echo "  평가 2/2: 96-case 전체 — G12_h8z 단독"
    echo "  비교: P2_LR2 / G12_h8z"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${P2_LR2_RECON_ROOT}:P2_LR2"                          \
            "${RECON_ROOT}/G12_h8z:G12_h8z"                         \
        --degraded-root "${DEGRADED_ROOT}"                           \
        --test-csv      "${TEST_CSV}"                                \
        --modalities    flair t1ce                                   \
        --output-csv    eval_G12_h8z_full96.csv

    if [ $? -eq 0 ]; then
        echo "  ✅ 96-case 평가 완료 → eval_G12_h8z_full96.csv"
        echo ""
        echo "  ── 비교 기준점 (96-case 전체) ──────────────────────────"
        echo "  no_gating: FLAIR 26.643 / T1CE 26.332 / 평균 26.487 dB"
        echo "  P2_LR2:    FLAIR 26.845 / T1CE 26.373 / 평균 26.609 dB"
        echo "  ── G11 파일럿(10명) 참고 ───────────────────────────────"
        echo "  G11_h8(G6, 10명): FLAIR 27.407 / T1CE 26.458 / 평균 26.932"
        echo "  ─────────────────────────────────────────────────────────"
        echo "  ── GPU2 합산 평가 실행 방법 (GPU2 완료 후) ─────────────"
        echo "  SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=2 ./run_G12_gpu2.sh"
        echo "  → eval_G12_pilot.csv / eval_G12_full96.csv 생성"
        echo "  ─────────────────────────────────────────────────────────"
    else
        echo "  ❌ 96-case 평가 실패"
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
echo "  실험 경로     : ${RECON_ROOT}/${EXP_TAG}"
echo "  파일럿 결과   : eval_G12_h8z_pilot.csv  (10명, G6 vs G7 비교)"
echo "  전체 결과     : eval_G12_h8z_full96.csv (96 cases)"
echo "  G12_h4(GPU2)  : hidden_dim=4, n-iter=2000, G6(no-z-gating)"
echo "  G12_h8z(GPU3) : hidden_dim=8, n-iter=2000, G7(z-gating ON)"
echo "  합산 평가     : GPU2에서 SKIP_TRAIN=1로 재실행"
echo "================================================================"

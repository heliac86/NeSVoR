#!/usr/bin/env bash
# ================================================================
# run_G8_pilot_sweep.sh  (v1 — 2026-05-26)
# G_C2 기반 파일럿 하이퍼파라미터 탐색 (20 cases)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# run_gating_sweep.sh 결과, G_C2(gating-grad-clip=0.1)가
# FLAIR/T1CE 양 모달리티에서 가장 안정적으로 우수한 성능을 보임.
#   G_C2 기준: FLAIR PSNR 26.763 / SSIM 0.913908 / LPIPS 0.026555
#              T1CE  PSNR 26.367 / SSIM 0.873795 / LPIPS 0.044451
#
# 가중치 분석 결과: 12개 레벨 중 L7에 평균 대비 1.4배 가중치 집중.
# → 모델이 중간~중고 주파수(조직 경계) 레벨을 자체 학습으로 선택.
# → grad-clip이 L7 과집중에 의한 gradient 폭발을 억제한 것으로 해석.
# → diversity loss 목표를 "균등 분화"보다 "자연 분화 보조"로 재설정 가능.
#
# ── 탐색 설계 ──────────────────────────────────────────────────
# 기준: G_C2 (div-w=0.08, div-v=0.06, g-clip=0.1, lr-scale=1.0, space=raw)
#
# 세트명      div-w  div-v  g-clip  lr-scale  div-space  target-var  탐색 목적
# --------    -----  -----  ------  --------  ---------  ----------  --------
# G_C2(기준) 0.08   0.06   0.1     1.0       raw        0.06        재사용(재실행 없음)
# P1_LR05    0.08   0.06   0.1     0.5       raw        0.06        lr-scale 낮춤 → gating 느리게 수렴
# P2_LR2     0.08   0.06   0.1     2.0       raw        0.06        lr-scale 높임 → gating 빠르게 분화
# P3_SM      0.08   0.06   0.1     1.0       softmax    0.50        div-space 전환 (스케일 상향 필요)
# P5_DW06    0.06   0.06   0.1     1.0       raw        0.06        div-w 낮춤 (clip 환경 최적 재탐색)
# P6_DW10    0.10   0.06   0.1     1.0       raw        0.06        div-w 높임 (clip 환경 최적 재탐색)
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q1. gating lr-scale 낮춤이 안정성을 개선하는가?    → G_C2 vs P1_LR05
# Q2. gating lr-scale 높임이 분화를 촉진하는가?     → G_C2 vs P2_LR2
# Q3. softmax 공간 diversity가 raw보다 효과적인가?   → G_C2 vs P3_SM
# Q4. clip 환경에서 div-w 감소가 유리한가?          → G_C2 vs P5_DW06
# Q5. clip 환경에서 div-w 증가가 유리한가?          → G_C2 vs P6_DW10
#
# ── 환자 선정 ──────────────────────────────────────────────────
# test.csv에서 seed=42 기반 무작위 10명 추출
# → 매 실행 시 동일한 10명 보장 (재현성 확보)
# 모달리티: flair, t1ce → 총 20 cases / 세트
#
# ── P3_SM 주의 사항 ──────────────────────────────────────────
# --diversity-loss-space softmax 전환 시:
#   softmax(level_weights)*n_levels의 variance 스케일은
#   raw logit variance와 완전히 다름 (parsers.py 주석 참조).
#   raw 기준 target=0.06 → softmax 기준 target=0.50으로 상향.
#   (softmax 출력 범위: 각 레벨 가중치 ∈ [0, n_levels], 균등 시 분산≈0)
#
# ── 판단 기준 ──────────────────────────────────────────────────
# FLAIR PSNR 기준 G_C2(26.763) 대비 +0.05 dB 이상 → 96-case full run 격상
# 양 모달리티 모두 개선 방향이면 후보 유지 (수치 미달이라도)
#
# ── 총 실행 ─────────────────────────────────────────────────
# 신규 5세트 × 20 cases = 100 케이스
# G_C2 기준 포함 6세트 비교
# 예상 총 시간: 신규 5세트 × ~1.5시간 ≈ 7~8시간
#
# ── 사용법 ──────────────────────────────────────────────────
#   chmod +x run_G8_pilot_sweep.sh
#   ./run_G8_pilot_sweep.sh                  # 학습 + 평가 전체
#   SKIP_TRAIN=1 ./run_G8_pilot_sweep.sh     # 평가만 재실행
#   SKIP_EVAL=1  ./run_G8_pilot_sweep.sh     # 학습만 실행
#   ENABLE_P1_LR05=0 ./run_G8_pilot_sweep.sh # P1_LR05 건너뜀
#   ENABLE_P2_LR2=0  ./run_G8_pilot_sweep.sh # P2_LR2  건너뜀
#   ENABLE_P3_SM=0   ./run_G8_pilot_sweep.sh # P3_SM   건너뜀
#   ENABLE_P5_DW06=0 ./run_G8_pilot_sweep.sh # P5_DW06 건너뜀
#   ENABLE_P6_DW10=0 ./run_G8_pilot_sweep.sh # P6_DW10 건너뜀
#
# ── 출력 파일 명명 규칙 ──────────────────────────────────────
#   recon_G8_pilot/{EXP_TAG}/BraTS20_Training_{PATIENT}/
#     {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_P1_LR05=${ENABLE_P1_LR05:-1}
ENABLE_P2_LR2=${ENABLE_P2_LR2:-1}
ENABLE_P3_SM=${ENABLE_P3_SM:-1}
ENABLE_P5_DW06=${ENABLE_P5_DW06:-1}
ENABLE_P6_DW10=${ENABLE_P6_DW10:-1}

# ── 실행 단계 플래그 ─────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_G8_pilot"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# G_C2 기준: run_gating_sweep.sh에서 생성된 결과 재사용
G_C2_RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_gating_sweep/G_C2"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── seed=42 기반 파일럿 환자 10명 무작위 추출 ────────────────
# 매 실행 시 동일한 10명이 선택됨 (재현성 보장)
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
echo "  파일럿 환자 목록 (seed=42, 10명):"
for ID in "${PILOT_IDS[@]}"; do
    echo "    - ${ID}"
done
echo "================================================================"

# ── BASE_ARGS: G_C2 확정 설정 재현 ──────────────────────────
# div-w, div-v, g-clip, lr-scale, div-space, target-var만 세트별로 덮어씀
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
# 세트 P1_LR05: gating-lr-scale=0.5
#
#   목적: gating 학습률을 절반으로 낮춰 느리고 안정적인 수렴 유도.
#         L7 집중 패턴이 이미 자연 발생하므로, 과도한 업데이트 억제가
#         다른 레벨의 적절한 기여를 보존하는지 확인.
#   변경: --gating-lr-scale 0.5
#   비교: G_C2 (lr-scale=1.0) vs P1_LR05 → lr-scale 낮춤 효과
# ================================================================
if [ "${ENABLE_P1_LR05}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P1_LR05: div-w=0.08, div-v=0.06, g-clip=0.1, lr-scale=0.5"
    echo "  비교: G_C2 (lr-scale=1.0) vs P1_LR05 → lr-scale 낮춤 효과"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P1_LR05" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.1  \
                --gating-lr-scale       0.5  \
                --diversity-loss-space  raw
        done
    done
else
    echo "  [SKIP] P1_LR05 (ENABLE_P1_LR05=0)"
fi

# ================================================================
# 세트 P2_LR2: gating-lr-scale=2.0
#
#   목적: gating 학습률을 2배로 높여 레벨 분화를 더 빠르게 촉진.
#         L7 집중이 더 선명하게 나타나는지, 혹은 과도한 분화로
#         다른 레벨이 억압되는지 확인.
#   변경: --gating-lr-scale 2.0
#   비교: G_C2 (lr-scale=1.0) vs P2_LR2 → lr-scale 높임 효과
#         P1_LR05 vs P2_LR2              → lr-scale 방향성 파악
# ================================================================
if [ "${ENABLE_P2_LR2}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P2_LR2: div-w=0.08, div-v=0.06, g-clip=0.1, lr-scale=2.0"
    echo "  비교: G_C2 (lr-scale=1.0) vs P2_LR2 → lr-scale 높임 효과"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P2_LR2" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.1  \
                --gating-lr-scale       2.0  \
                --diversity-loss-space  raw
        done
    done
else
    echo "  [SKIP] P2_LR2 (ENABLE_P2_LR2=0)"
fi

# ================================================================
# 세트 P3_SM: diversity-loss-space=softmax, target-diversity-var=0.50
#
#   목적: diversity loss를 raw logit 공간이 아닌 실제 gating 출력
#         (softmax * n_levels) 공간에서 계산.
#         grad-clip이 logit 절대값을 억제하면 raw variance의 의미가
#         약해지므로, 실제 작동하는 가중치 기준으로 전환.
#
#   ⚠ target-var 스케일 주의:
#         raw logit 기준: target=0.06 (logit 값 자체의 분산)
#         softmax 출력 기준: 균등 시 각 레벨 가중치=1.0, 분산≈0에 가까움.
#         L7에 1.4배 집중된 실측 패턴 기준 추정 분산 ≈ 0.10~0.20.
#         → 목표를 0.50으로 설정: 자연 패턴보다 높은 분화를 유도하되
#           과도한 집중은 억제 (relu(target - var)이므로 var≥0.50이면 loss=0)
#
#   변경: --diversity-loss-space softmax / --target-diversity-var 0.50
#   비교: G_C2 (raw, 0.06) vs P3_SM (softmax, 0.50)
# ================================================================
if [ "${ENABLE_P3_SM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P3_SM: div-w=0.08, div-v=0.06, g-clip=0.1, lr-scale=1.0"
    echo "              div-space=softmax, target-var=0.50"
    echo "  비교: G_C2 (raw, 0.06) vs P3_SM → diversity 공간 전환 효과"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P3_SM" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var  0.50 \
                --gating-grad-clip      0.1  \
                --gating-lr-scale       1.0  \
                --diversity-loss-space  softmax
        done
    done
else
    echo "  [SKIP] P3_SM (ENABLE_P3_SM=0)"
fi

# ================================================================
# 세트 P5_DW06: weight-diversity-loss=0.06
#
#   목적: grad-clip=0.1 환경에서 diversity loss 강도를 낮춘 효과 확인.
#         clip이 gradient를 이미 제약하므로, div-w를 함께 낮추면
#         gating에 가해지는 총 압력(clip + loss)이 적절히 균형잡힐 수 있음.
#         L7 자연 집중 패턴을 과도하게 방해하지 않는 최소 필요 강도 탐색.
#   변경: --weight-diversity-loss 0.06
#   비교: G_C2 (div-w=0.08) vs P5_DW06 → div-w 감소 효과
# ================================================================
if [ "${ENABLE_P5_DW06}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P5_DW06: div-w=0.06, div-v=0.06, g-clip=0.1, lr-scale=1.0"
    echo "  비교: G_C2 (div-w=0.08) vs P5_DW06 → div-w 감소 효과"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P5_DW06" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.06 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.1  \
                --gating-lr-scale       1.0  \
                --diversity-loss-space  raw
        done
    done
else
    echo "  [SKIP] P5_DW06 (ENABLE_P5_DW06=0)"
fi

# ================================================================
# 세트 P6_DW10: weight-diversity-loss=0.10
#
#   목적: grad-clip=0.1 환경에서 diversity loss 강도를 높인 효과 확인.
#         clip 없이 div-w=0.12였던 G_DH보다 더 적은 값(0.10)으로도
#         clip 덕분에 안정적 분화가 가능한지 탐색.
#         G_DH(clip=0.0, div-w=0.12)와 대응 비교 포인트.
#   변경: --weight-diversity-loss 0.10
#   비교: G_C2 (div-w=0.08) vs P6_DW10 → div-w 증가 효과
#         G_DH (clip=0.0, div-w=0.12)  vs P6_DW10 → clip과 div-w 교호 효과
# ================================================================
if [ "${ENABLE_P6_DW10}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 P6_DW10: div-w=0.10, div-v=0.06, g-clip=0.1, lr-scale=1.0"
    echo "  비교: G_C2 (div-w=0.08) vs P6_DW10 → div-w 증가 효과"
    echo "  케이스: 20  예상 시간: ~1.5시간"
    echo "################################################################"
    for FULL_ID in "${PILOT_IDS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "P6_DW10" "${FULL_ID}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.10 \
                --target-diversity-var  0.06 \
                --gating-grad-clip      0.1  \
                --gating-lr-scale       1.0  \
                --diversity-loss-space  raw
        done
    done
else
    echo "  [SKIP] P6_DW10 (ENABLE_P6_DW10=0)"
fi

fi  # end SKIP_TRAIN

# ================================================================
# 평가 실행
# G_C2 기준: recon_gating_sweep/G_C2/ (기존 결과 재사용, 재실행 없음)
# 단, G_C2의 경우 파일럿 10명에 해당하는 케이스만 필터링하여 비교
# → evaluate_batch_v2.py의 --test-csv를 파일럿 CSV로 교체
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then

    # 파일럿 환자 ID를 임시 CSV로 저장 (헤더 포함)
    PILOT_CSV="/tmp/pilot_test_G8.csv"
    echo "Brats20ID" > "${PILOT_CSV}"
    for ID in "${PILOT_IDS[@]}"; do
        echo "${ID}" >> "${PILOT_CSV}"
    done

    echo ""
    echo "################################################################"
    echo "  평가 시작: evaluate_batch_v2.py"
    echo "  총 6세트: G_C2(기준) / P1_LR05 / P2_LR2 / P3_SM / P5_DW06 / P6_DW10"
    echo "  파일럿 CSV: ${PILOT_CSV} (10명 × 2 모달 = 20 cases)"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    /dshome/ddualab/conda_env/nerf_tcnn/bin/python evaluate_batch_v2.py \
        --recon-dirs \
            "${G_C2_RECON_ROOT}:G_C2"               \
            "${RECON_ROOT}/P1_LR05:P1_LR05"         \
            "${RECON_ROOT}/P2_LR2:P2_LR2"           \
            "${RECON_ROOT}/P3_SM:P3_SM"             \
            "${RECON_ROOT}/P5_DW06:P5_DW06"         \
            "${RECON_ROOT}/P6_DW10:P6_DW10"         \
        --degraded-root "${DEGRADED_ROOT}"           \
        --test-csv      "${PILOT_CSV}"               \
        --modalities    flair t1ce                   \
        --output-csv    eval_G8_pilot_sweep.csv

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_G8_pilot_sweep.csv"
        echo ""
        echo "  ── 다음 단계 판단 기준 ──────────────────────────────"
        echo "  G_C2 FLAIR PSNR 기준점: 26.763 dB (20-case 파일럿 기준으로 변동 가능)"
        echo "  +0.05 dB 이상 개선된 세트 → 96-case full run 격상"
        echo "  양 모달리티 모두 개선 방향 → 후보 유지 (수치 미달이라도)"
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
echo "  G_C2 경로   : ${G_C2_RECON_ROOT}"
echo "  파일럿 경로 : ${RECON_ROOT}"
echo "  결과 CSV    : eval_G8_pilot_sweep.csv"
echo "  파일럿 CSV  : /tmp/pilot_test_G8.csv (임시 파일, 재실행 시 자동 재생성)"
echo "================================================================"

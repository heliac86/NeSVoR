#!/usr/bin/env bash
# ================================================================
# run_optionB_sweep.sh
# Option B (Soft Reweighting HSM) beta 탐색
#
# ── 변경 사항 (vs run_optionA_sweep.sh) ──────────────────────────
#   Option A : per-batch multinomial → 쉬운 슬라이스 배제 → 붕괴
#   Option B : 에포크-셔플 구조 유지 + 균등/HSM 혼합 가중치
#              모든 슬라이스가 에포크당 정확히 1회 등장 보장
#              --hsm-beta 로 혼합 비율 제어
#
# ── alpha 고정 ──────────────────────────────────────────────────
#   alpha=0.5 고정 (이전 전체 실험 최고 성능: hsm_O)
#
# ── 탐색 세트 ───────────────────────────────────────────────────
#   optB_A : beta=0.5  (균등 50% + HSM 50%)
#   optB_B : beta=0.3  (균등 70% + HSM 30%)
#   optB_C : beta=0.1  (균등 90% + HSM 10%, 가장 약한 편향)
#
#   beta=1.0 은 기존 hsm_O 동작과 동일 → 새 탐색 불필요
#
# ── 예상 시간 ────────────────────────────────────────────────────
#   세트당 : 48명 × 2 모달리티 × ~260초 ≈ 6.9시간
#   3세트 합계 : 약 20.7시간
#
# ── 출력 구조 ────────────────────────────────────────────────────
#   recon_HSM_optB_A/
#     BraTS20_Training_{ID}/
#       {ID}_flair_4x5_optB_A.nii.gz
#       {ID}_t1ce_4x5_optB_A.nii.gz
#
# ── 사용법 ──────────────────────────────────────────────────────
#   chmod +x run_optionB_sweep.sh
#   ./run_optionB_sweep.sh                     # 전체 3세트 순차 실행
#   ENABLE_OPT_A=0 ./run_optionB_sweep.sh     # optB_A 건너뜀
#   ENABLE_OPT_B=0 ./run_optionB_sweep.sh     # optB_B 건너뜀
#   ENABLE_OPT_C=0 ./run_optionB_sweep.sh     # optB_C 건너뜀
#   SKIP_EVAL=1    ./run_optionB_sweep.sh     # 평가 건너뜀
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────────
ENABLE_OPT_A=${ENABLE_OPT_A:-1}
ENABLE_OPT_B=${ENABLE_OPT_B:-1}
ENABLE_OPT_C=${ENABLE_OPT_C:-1}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ────────────────────────────────────────────────────
RECON_BASE="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="${RECON_BASE}/test.csv"

# ── 공통 학습 설정 ───────────────────────────────────────────────
# alpha=0.5 고정 (이전 전체 실험 최고 성능 기준)
# start_iter=0 고정 (run_hsm_dataset.sh 전통 유지)
INTENSITY_MEAN=308
MODALITIES=(flair t1ce)
ALPHA="0.5"

BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-image 1.0
  --delta 0.05
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 0.0
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
  --hard-mining-start-iter 0
  --slice-residual-alpha ${ALPHA}
"

# ── 전체 실패 누적 ───────────────────────────────────────────────
ALL_FAILED=()

# ================================================================
# 세트 실행 함수
#   $1 = TAG   (예: optB_A)
#   $2 = BETA  (예: 0.5)
# ================================================================
run_dataset_set() {
    local TAG=$1
    local BETA=$2

    local RECON_ROOT="${RECON_BASE}/recon_HSM_${TAG}"
    local FAILED_CASES=()
    local TOTAL=0
    local SUCCESS=0

    echo ""
    echo "################################################################"
    echo "  세트 시작: ${TAG}  (alpha=${ALPHA}, beta=${BETA}, start_iter=0, Option B)"
    echo "  출력 루트: ${RECON_ROOT}"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    mkdir -p "${RECON_ROOT}"

    echo ""
    echo "  [단계 2] test.csv 기반 순차 학습 (alpha=${ALPHA}, beta=${BETA}, start_iter=0)"

    mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

    for FULL_ID in "${FULL_IDS[@]}"; do
        PATIENT="${FULL_ID##*_}"
        PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
        mkdir -p "${PATIENT_DIR}"

        for MODALITY in "${MODALITIES[@]}"; do
            DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${TAG}.nii.gz"

            # 이미 존재하면 스킵
            if [ -f "${DST}" ]; then
                echo "    [SKIP] 이미 존재: ${PATIENT} / ${MODALITY}"
                continue
            fi

            INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

            if [ ! -f "${INPUT}" ]; then
                echo "    ❌ 입력 없음: ${PATIENT} / ${MODALITY}  (${INPUT})"
                FAILED_CASES+=("${TAG}_${PATIENT}_${MODALITY}_input_missing")
                TOTAL=$((TOTAL + 1))
                continue
            fi

            echo ""
            echo "  ================================================================"
            echo "  [START] ${TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            echo "  ================================================================"

            # shellcheck disable=SC2086
            nesvor reconstruct \
                --input-stacks  "${INPUT}" \
                --output-volume "${DST}" \
                --sample-mask   "${INPUT}" \
                --output-intensity-mean ${INTENSITY_MEAN} \
                ${BASE_ARGS} \
                --hsm-beta "${BETA}"

            EXIT_CODE=$?
            TOTAL=$((TOTAL + 1))
            if [ ${EXIT_CODE} -ne 0 ]; then
                echo "  ❌ FAILED: ${TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
                FAILED_CASES+=("${TAG}_${PATIENT}_${MODALITY}")
            else
                SUCCESS=$((SUCCESS + 1))
                echo "  ✅ DONE : ${TAG} / ${PATIENT} / ${MODALITY}"
                echo "       → ${DST}"
                echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            fi
        done
    done

    # 세트 요약
    echo ""
    echo "  ── 세트 ${TAG} 완료: $(date '+%Y-%m-%d %H:%M:%S') ──"
    echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
    if [ ${#FAILED_CASES[@]} -gt 0 ]; then
        echo "  ❌ 실패 (${#FAILED_CASES[@]}건):"
        for c in "${FAILED_CASES[@]}"; do
            echo "       - ${c}"
        done
        ALL_FAILED+=("${FAILED_CASES[@]}")
    else
        echo "  ✅ 모든 케이스 성공"
    fi
}

# ================================================================
# 세트별 순차 실행
# ================================================================

if [ "${ENABLE_OPT_A}" -eq 1 ]; then
    run_dataset_set "optB_A" "0.5"
else
    echo "  [SKIP] optB_A (ENABLE_OPT_A=0)"
fi

if [ "${ENABLE_OPT_B}" -eq 1 ]; then
    run_dataset_set "optB_B" "0.3"
else
    echo "  [SKIP] optB_B (ENABLE_OPT_B=0)"
fi

if [ "${ENABLE_OPT_C}" -eq 1 ]; then
    run_dataset_set "optB_C" "0.1"
else
    echo "  [SKIP] optB_C (ENABLE_OPT_C=0)"
fi

# ================================================================
# 평가 실행
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  평가 시작: eval_hsm_dataset.py"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    python eval_hsm_dataset.py

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_results/eval_hsm_dataset.csv"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 전체 최종 요약 ────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${#ALL_FAILED[@]} -eq 0 ]; then
    echo "  ✅ 모든 세트 성공"
else
    echo "  ❌ 전체 실패 케이스 (${#ALL_FAILED[@]}건):"
    for c in "${ALL_FAILED[@]}"; do
        echo "       - ${c}"
    done
fi
echo "================================================================"

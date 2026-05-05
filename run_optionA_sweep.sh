#!/usr/bin/env bash
# ================================================================
# run_optionA_sweep.sh
# Option A (per-batch multinomial HSM) alpha 탐색
#
# ── 탐색 세트 ─────────────────────────────────────────────────────
#   optA_Q : alpha=0.9  (느린 EMA, 안정적)
#   optA_R : alpha=0.7  (중간)
#   optA_S : alpha=0.5  (빠른 EMA, 기존 hsm_O 동일 alpha)
#
# ── 기존 hsm_O와의 차이점 ──────────────────────────────────
#   기존 hsm_O : get_batch() 에포크-셔플 구조 → 픽셀 빈도 차이 없음
#   이 스크립트 : per-batch multinomial → 잔차 높은 슬라이스 실제로 더 자주 샘플링
#
# ── 단계 2만 실행 (단계 1 기존 5명 복사 없음) ─────────────────
#   test.csv에 정의된 전체 환자 48명 전원 신규 학습
#
# ── 예상 시간 ────────────────────────────────────────────────
#   세트당 : 48명 × 2 모달리티 × ~260슈 ≈ 6.9시간
#   3세트 합계 : 약 20.7시간
#
# ── 출력 구조 ────────────────────────────────────────────────
#   recon_HSM_optA_Q/
#     BraTS20_Training_{ID}/
#       {ID}_flair_4x5_optA_Q.nii.gz
#       {ID}_t1ce_4x5_optA_Q.nii.gz
#
# ── 사용법 ─────────────────────────────────────────────────
#   chmod +x run_optionA_sweep.sh
#   ./run_optionA_sweep.sh                     # 전체 3세트 순차 실행
#   ENABLE_OPT_Q=0 ./run_optionA_sweep.sh     # optA_Q 건너뛰
#   ENABLE_OPT_R=0 ./run_optionA_sweep.sh     # optA_R 건너뛰
#   ENABLE_OPT_S=0 ./run_optionA_sweep.sh     # optA_S 건너뛰
#   SKIP_EVAL=1    ./run_optionA_sweep.sh     # 평가 건너뛰
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────
ENABLE_OPT_Q=${ENABLE_OPT_Q:-1}
ENABLE_OPT_R=${ENABLE_OPT_R:-1}
ENABLE_OPT_S=${ENABLE_OPT_S:-1}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_BASE="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="${RECON_BASE}/test.csv"

# ── 공통 학습 설정 (hsm_O와 동일, hard-mining-start-iter=0 고정) ───
INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

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
"

# ── 전체 실패 누적 ─────────────────────────────────────────
ALL_FAILED=()

# ================================================================
# 세트 실행 함수 (단계 2만)
#   $1 = TAG  (예: optA_Q)
#   $2 = ALPHA (예: 0.9)
# ================================================================
run_dataset_set() {
    local TAG=$1
    local ALPHA=$2

    local RECON_ROOT="${RECON_BASE}/recon_HSM_${TAG}"
    local FAILED_CASES=()
    local TOTAL=0
    local SUCCESS=0

    echo ""
    echo "################################################################"
    echo "  세트 시작: ${TAG}  (alpha=${ALPHA}, start_iter=0, Option A)"
    echo "  출력 루트: ${RECON_ROOT}"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    mkdir -p "${RECON_ROOT}"

    # ── 단계 2: test.csv 기반 순차 학습 ─────────────────────────
    echo ""
    echo "  [단계 2] test.csv 기반 순차 학습 (alpha=${ALPHA}, start_iter=0)"

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
                --slice-residual-alpha "${ALPHA}"

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

if [ "${ENABLE_OPT_Q}" -eq 1 ]; then
    run_dataset_set "optA_Q" "0.9"
else
    echo "  [SKIP] optA_Q (ENABLE_OPT_Q=0)"
fi

if [ "${ENABLE_OPT_R}" -eq 1 ]; then
    run_dataset_set "optA_R" "0.7"
else
    echo "  [SKIP] optA_R (ENABLE_OPT_R=0)"
fi

if [ "${ENABLE_OPT_S}" -eq 1 ]; then
    run_dataset_set "optA_S" "0.5"
else
    echo "  [SKIP] optA_S (ENABLE_OPT_S=0)"
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

# ── 전체 최종 요약 ────────────────────────────────────────────
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

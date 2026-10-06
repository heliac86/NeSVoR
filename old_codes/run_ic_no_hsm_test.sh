#!/usr/bin/env bash
# ================================================================
# run_ic_no_hsm_test.sh
# IC 브랜치 설정에서 HSM(하드마이닝)만 제거한 어블레이션 테스트
#
# run_ic_repro_test.sh 대비 변경사항:
#   - BASE_ARGS에서 --hard-mining-main-loss 제거
#   - 출력 디렉토리 : recon_IC_no_hsm/
#   - 평가 결과 CSV  : eval_results/ic_no_hsm_test.csv
#   - Ours_orig(recon_IC) vs No_HSM(recon_IC_no_hsm) 비교
#
# 사용법:
#   chmod +x run_ic_no_hsm_test.sh
#   ./run_ic_no_hsm_test.sh              # 학습 + 평가 전체 실행
#   SKIP_TRAIN=1 ./run_ic_no_hsm_test.sh # 학습 스킵, 평가만 재실행
#   SKIP_EVAL=1  ./run_ic_no_hsm_test.sh # 평가 스킵, 학습만 실행
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

TEST_PATIENTS=(
    "BraTS20_Training_003"
    "BraTS20_Training_125"
    "BraTS20_Training_356"
)

MODALITIES=(flair t1ce)

NESVOR_ROOT="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

# [DIFF] 출력 경로만 다름
REPRO_ROOT="${NESVOR_ROOT}/recon_IC_no_hsm"
ORIG_ROOT="${NESVOR_ROOT}/recon_IC"
EVAL_DIR="${NESVOR_ROOT}/eval_results"
RESULT_CSV="${EVAL_DIR}/ic_no_hsm_test.csv"

NERF_TCNN_ENV="/dshome/ddualab/conda_env/nerf_tcnn"
INTENSITY_MEAN=308

# [DIFF] --hard-mining-main-loss 제거
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-image 1.0
  --delta 0.05
  --gating-grad-clip 0.0
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# ────────────────────────────────────────────────────────────────
# 단계 1: 학습
# ────────────────────────────────────────────────────────────────
TOTAL=0
SUCCESS=0
declare -a FAILED_CASES=()

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  [단계 1] IC No-HSM 학습 시작"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    for FULL_ID in "${TEST_PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            TOTAL=$((TOTAL + 1))
            PATIENT="${FULL_ID##*_}"

            PATIENT_DIR="${REPRO_ROOT}/${FULL_ID}"
            DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_IC_no_hsm.nii.gz"
            DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_IC_no_hsm.pt"
            INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

            mkdir -p "${PATIENT_DIR}"

            if [ -f "${DST}" ]; then
                echo "  [SKIP] 이미 존재: ${FULL_ID} / ${MODALITY}"
                SUCCESS=$((SUCCESS + 1))
                continue
            fi

            if [ ! -f "${INPUT}" ]; then
                echo "  ❌ 입력 파일 없음: ${FULL_ID} / ${MODALITY} → ${INPUT}"
                FAILED_CASES+=("${FULL_ID}_${MODALITY}")
                continue
            fi

            echo ""
            echo "  ================================================================"
            echo "  [TRAIN] Patient=${PATIENT}  Modality=${MODALITY}  (No HSM)"
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            echo "  ================================================================"

            # shellcheck disable=SC2086
            nesvor reconstruct \
                --input-stacks  "${INPUT}" \
                --output-volume "${DST}" \
                --output-model  "${DST_PT}" \
                --sample-mask   "${INPUT}" \
                --output-intensity-mean ${INTENSITY_MEAN} \
                ${BASE_ARGS}

            EXIT_CODE=$?
            if [ ${EXIT_CODE} -eq 0 ]; then
                SUCCESS=$((SUCCESS + 1))
                echo "  ✅ 완료: ${FULL_ID} / ${MODALITY}"
            else
                FAILED_CASES+=("${FULL_ID}_${MODALITY}")
                echo "  ❌ 실패 (exit=${EXIT_CODE}): ${FULL_ID} / ${MODALITY}"
            fi
        done
    done

    echo ""
    echo "  ── 학습 요약 ──────────────────────────────────────────────"
    echo "  성공: ${SUCCESS} / ${TOTAL}"
    if [ ${#FAILED_CASES[@]} -gt 0 ]; then
        echo "  ❌ 실패한 케이스:"
        for c in "${FAILED_CASES[@]}"; do echo "       - ${c}"; done
    else
        echo "  ✅ 모든 케이스 학습 완료"
    fi
else
    echo "  [SKIP] 학습 (SKIP_TRAIN=1)"
fi

# ────────────────────────────────────────────────────────────────
# 단계 2: 평가
# ────────────────────────────────────────────────────────────────
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  [다단계 2] 이미지 품질 평가 시작"
    echo "  비교: Ours_orig(recon_IC) vs No_HSM(recon_IC_no_hsm)"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    mkdir -p "${EVAL_DIR}"

    TEMP_CSV="${EVAL_DIR}/_no_hsm_test_patients.csv"
    echo "Brats20ID" > "${TEMP_CSV}"
    for FULL_ID in "${TEST_PATIENTS[@]}"; do
        echo "${FULL_ID}" >> "${TEMP_CSV}"
    done
    echo "  [INFO] 임시 평가용 CSV 생성: ${TEMP_CSV}"

    TEMP_EVAL="${EVAL_DIR}/_ic_no_hsm_eval.py"
    cat > "${TEMP_EVAL}" << 'PYEOF'
"""
_ic_no_hsm_eval.py  —  evaluate_batch_v2.py 래퍼
IC No-HSM 어블레이션: Ours_orig (recon_IC) vs No_HSM (recon_IC_no_hsm) 비교.
"""
import importlib.util, os

NESVOR_ROOT = os.environ["NESVOR_ROOT"]
RESULT_CSV  = os.environ["RESULT_CSV"]
TEMP_CSV    = os.environ["TEMP_CSV"]
REPRO_ROOT  = os.environ["REPRO_ROOT"]
ORIG_ROOT   = os.environ["ORIG_ROOT"]

spec = importlib.util.spec_from_file_location(
    "evaluate_batch_v2",
    os.path.join(NESVOR_ROOT, "evaluate_batch_v2.py"),
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.TEST_CSV   = TEMP_CSV
m.RESULT_CSV = RESULT_CSV
m.METHODS    = {
    "Ours_orig" : ORIG_ROOT,   # IC 최고성능 (기준)
    "No_HSM"    : REPRO_ROOT,  # HSM 제거 버전 (비교 대상)
}

m.main()
PYEOF

    echo "  [INFO] 임시 평가 스크립트 생성: ${TEMP_EVAL}"

    export NESVOR_ROOT RESULT_CSV TEMP_CSV REPRO_ROOT ORIG_ROOT
    conda run \
        -p "${NERF_TCNN_ENV}" \
        --no-capture-output \
        python "${TEMP_EVAL}"

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → ${RESULT_CSV}"
    fi

    if [ ${EVAL_EXIT} -eq 0 ]; then
        rm -f "${TEMP_CSV}" "${TEMP_EVAL}"
        echo "  [CLEAN] 임시 파일 삭제 완료"
    else
        echo "  [INFO] 임시 파일 보존 (디버깅용): ${TEMP_CSV}, ${TEMP_EVAL}"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ────────────────────────────────────────────────────────────────
# 최종 요약
# ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  IC No-HSM 어블레이션 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
fi
echo "  No-HSM 결과 경로: ${REPRO_ROOT}"
echo "  평가 결과 CSV : ${RESULT_CSV}"
echo "================================================================"

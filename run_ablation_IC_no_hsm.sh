#!/usr/bin/env bash
# ================================================================
# run_ablation_IC_no_hsm.sh
# IC 브랜치 어빌레이션: Ours / No_FF / No_Gating
# 공통 조건: HSM 미사용
#
# 출력 구조:
#   recon_ablation_re_IC_Ours/
#   recon_ablation_re_IC_No_FF/
#   recon_ablation_re_IC_No_Gating/
#
# 사용법:
#   chmod +x run_ablation_IC_no_hsm.sh
#   ./run_ablation_IC_no_hsm.sh              # 학습 + 평가 전체
#   SKIP_TRAIN=1 ./run_ablation_IC_no_hsm.sh # 평가만 재실행
#   SKIP_EVAL=1  ./run_ablation_IC_no_hsm.sh # 학습만 실행
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ────────────────────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────────────────────
NESVOR_ROOT="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
RECON_BASE="/dshome/ddualab/dongnyeok/NeSVoR/recon_ablation_re_IC"

NERF_TCNN_ENV="/dshome/ddualab/conda_env/nerf_tcnn"
EVAL_DIR="${NESVOR_ROOT}/eval_results"
RESULT_CSV="${EVAL_DIR}/ablation_no_hsm.csv"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ────────────────────────────────────────────────────────────────
# 어빌레이션 조건 정의
# 형식: "추가플래그"  (빈 문자열 = BASE_ARGS 그대로)
# 폴더: ${RECON_BASE}_${COND_KEY}
# 파일 suffix: ${COND_KEY_LOWER}  (소문자 변환)
# ────────────────────────────────────────────────────────────────
CONDITION_KEYS=("Ours" "No_FF" "No_Gating")

declare -A EXTRA_FLAGS
EXTRA_FLAGS["Ours"]=""
EXTRA_FLAGS["No_FF"]="--weight-ff-loss 0.0"
EXTRA_FLAGS["No_Gating"]="--no-gating --weight-diversity-loss 0.0"

# ────────────────────────────────────────────────────────────────
# 공통 BASE_ARGS (HSM 미사용)
# ────────────────────────────────────────────────────────────────
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
    echo "  [단계 1] 어빌레이션 학습 시작"
    echo "  조건: ${CONDITION_KEYS[*]}"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    for COND_KEY in "${CONDITION_KEYS[@]}"; do
        RECON_ROOT="${RECON_BASE}_${COND_KEY}"
        SUFFIX="IC_no_hsm_$(echo "${COND_KEY}" | tr '[:upper:]' '[:lower:]')"
        FLAGS="${EXTRA_FLAGS[$COND_KEY]}"

        echo ""
        echo "  ================================================================"
        echo "  [COND] ${COND_KEY}"
        echo "  출력 루트 : ${RECON_ROOT}"
        echo "  ================================================================"

        while IFS= read -r FULL_ID; do
            [ -z "${FULL_ID}" ] && continue
            [ "${FULL_ID}" = "Brats20ID" ] && continue

            for MODALITY in "${MODALITIES[@]}"; do
                TOTAL=$((TOTAL + 1))
                PATIENT="${FULL_ID##*_}"

                PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
                DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${SUFFIX}.nii.gz"
                DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${SUFFIX}.pt"
                INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

                mkdir -p "${PATIENT_DIR}"

                if [ -f "${DST}" ]; then
                    echo "    [SKIP] 이미 존재: ${FULL_ID} / ${MODALITY} / ${COND_KEY}"
                    SUCCESS=$((SUCCESS + 1))
                    continue
                fi

                if [ ! -f "${INPUT}" ]; then
                    echo "    ❌ 입력 파일 없음: ${INPUT}"
                    FAILED_CASES+=("${COND_KEY}_${FULL_ID}_${MODALITY}")
                    continue
                fi

                echo ""
                echo "    [TRAIN] ${COND_KEY} | Patient=${PATIENT} | Modality=${MODALITY}"
                echo "    Time  : $(date '+%Y-%m-%d %H:%M:%S')"

                # shellcheck disable=SC2086
                nesvor reconstruct \
                    --input-stacks  "${INPUT}" \
                    --output-volume "${DST}" \
                    --output-model  "${DST_PT}" \
                    --sample-mask   "${INPUT}" \
                    --output-intensity-mean ${INTENSITY_MEAN} \
                    ${BASE_ARGS} \
                    ${FLAGS}

                EXIT_CODE=$?
                if [ ${EXIT_CODE} -eq 0 ]; then
                    SUCCESS=$((SUCCESS + 1))
                    echo "    ✅ 완료: ${COND_KEY} / ${FULL_ID} / ${MODALITY}"
                else
                    FAILED_CASES+=("${COND_KEY}_${FULL_ID}_${MODALITY}")
                    echo "    ❌ 실패 (exit=${EXIT_CODE}): ${COND_KEY} / ${FULL_ID} / ${MODALITY}"
                fi

            done
        done < "${TEST_CSV}"
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
    echo "  [단계 2] 이미지 품질 평가 시작"
    echo "  비교: Ours / No_FF / No_Gating"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    mkdir -p "${EVAL_DIR}"

    TEMP_EVAL="${EVAL_DIR}/_ablation_no_hsm_eval.py"
    cat > "${TEMP_EVAL}" << PYEOF
"""
_ablation_no_hsm_eval.py  —  evaluate_batch_v2.py 래퍼
어빌레이션: Ours / No_FF / No_Gating 비교 (HSM 미사용 공통)
"""
import importlib.util, os

NESVOR_ROOT = os.environ["NESVOR_ROOT"]
RESULT_CSV  = os.environ["RESULT_CSV"]
TEST_CSV    = os.environ["TEST_CSV"]
RECON_BASE  = os.environ["RECON_BASE"]

spec = importlib.util.spec_from_file_location(
    "evaluate_batch_v2",
    os.path.join(NESVOR_ROOT, "evaluate_batch_v2.py"),
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.TEST_CSV   = TEST_CSV
m.RESULT_CSV = RESULT_CSV
m.METHODS    = {
    "Ours"      : f"{RECON_BASE}_Ours",
    "No_FF"     : f"{RECON_BASE}_No_FF",
    "No_Gating" : f"{RECON_BASE}_No_Gating",
}
m.SUFFIXES = {
    "Ours"      : "IC_no_hsm_ours",
    "No_FF"     : "IC_no_hsm_no_ff",
    "No_Gating" : "IC_no_hsm_no_gating",
}

m.main()
PYEOF

    echo "  [INFO] 임시 평가 스크립트 생성: ${TEMP_EVAL}"

    export NESVOR_ROOT RESULT_CSV TEST_CSV RECON_BASE
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
        rm -f "${TEMP_EVAL}"
        echo "  [CLEAN] 임시 파일 삭제 완료"
    else
        echo "  [INFO] 임시 파일 보존 (디버깅용): ${TEMP_EVAL}"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ────────────────────────────────────────────────────────────────
# 최종 요약
# ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  어빌레이션 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
fi
for COND_KEY in "${CONDITION_KEYS[@]}"; do
    echo "  ${COND_KEY} 결과: ${RECON_BASE}_${COND_KEY}/"
done
echo "  평가 결과: ${RESULT_CSV}"
echo "================================================================"

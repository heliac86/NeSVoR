#!/usr/bin/env bash
# ================================================================
# run_ic_repro_test.sh
# IC 브랜치 재현 실험 스크립트 (3명 샘플링 버전)
#
# ── 목적 ────────────────────────────────────────────────────────
#   IC 브랜치(커밋 756dd67 기준) 코드가 과거 최고 성능 결과물
#   (recon_IC, 48명)을 재현하는지 3명 샘플 테스트로 검증한다.
#
# ── 재현 전략 ────────────────────────────────────────────────────
#   - 환자 3명(TEST_PATIENTS 배열)의 flair, t1ce 모달리티를
#     IC 브랜치 인자 세트로 새로 학습
#   - 출력은 recon_IC_repro/ 에 저장하여 원본 recon_IC/ 와 분리
#   - 학습 완료 후 evaluate_batch_v2.py를 통해
#       "Ours_orig(원본 recon_IC)"와 "Repro(재현 결과)"를 직접 비교
#   - 비교 대상 3명만 평가하여 속도 최적화 (TEST_CSV 임시 생성)
#
# ── 환자 선정 기준 ───────────────────────────────────────────────
#   원본 스크립트의 '기존 5명 복사' 대상(003/026/030/040/060)과
#   무관하게, 전체 48명 중 대표성 있는 3명을 선택:
#     BraTS20_Training_003   (복사 대상이었으나 재현 테스트로 재활용)
#     BraTS20_Training_126   (중간 번호)
#     BraTS20_Training_354   (후반 번호)
#   → 다른 환자로 교체하려면 TEST_PATIENTS 배열만 수정할 것
#
# ── 주요 인자 (run_ic_dataset.sh 동일 세트) ─────────────────────
#   n-iter=2000, batch=4096, lr=5e-3
#   weight-ff-loss=0.05, ff-alpha=2.0
#   weight-diversity-loss=0.08, target-diversity-var=0.06
#   diversity-loss-fn=variance, diversity-loss-space=raw
#   hard-mining-main-loss ON, gating-grad-clip=0.0
#   patch-size=48, n-patches=8
#   gamma=0.33, milestones=0.5 0.75 0.9, seed=42
#
# ── 사용법 ──────────────────────────────────────────────────────
#   chmod +x run_ic_repro_test.sh
#   ./run_ic_repro_test.sh              # 학습 + 평가 전체 실행
#   SKIP_TRAIN=1 ./run_ic_repro_test.sh # 학습 스킵, 평가만 재실행
#   SKIP_EVAL=1  ./run_ic_repro_test.sh # 평가 스킵, 학습만 실행
#
# ── 출력 ────────────────────────────────────────────────────────
#   recon_IC_repro/
#     BraTS20_Training_003/
#       003_flair_4x5_IC_repro.nii.gz
#       003_flair_4x5_IC_repro.pt
#       003_t1ce_4x5_IC_repro.nii.gz
#       003_t1ce_4x5_IC_repro.pt
#     BraTS20_Training_126/  ...
#     BraTS20_Training_354/  ...
#   eval_results/ic_repro_test.csv
# ================================================================

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 실행 플래그 ────────────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 환자 목록 ─────────────────────────────────────────────────
# 재현 테스트 대상 3명. 다른 환자로 교체 시 아래 배열만 수정.
TEST_PATIENTS=(
    "BraTS20_Training_003"
    "BraTS20_Training_126"
    "BraTS20_Training_354"
)

MODALITIES=(flair t1ce)

# ── 경로 설정 ─────────────────────────────────────────────────
NESVOR_ROOT="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

REPRO_ROOT="${NESVOR_ROOT}/recon_IC_repro"    # 재현 결과 저장
ORIG_ROOT="${NESVOR_ROOT}/recon_IC"           # 원본(최고성능) 결과
EVAL_DIR="${NESVOR_ROOT}/eval_results"
RESULT_CSV="${EVAL_DIR}/ic_repro_test.csv"

# evaluate_batch_v2.py 경로 (IC 브랜치 루트)
EVAL_SCRIPT="${NESVOR_ROOT}/evaluate_batch_v2.py"

INTENSITY_MEAN=308

# ── BASE ARGS (run_ic_dataset.sh 와 100% 동일) ────────────────
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
  --hard-mining-main-loss
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
declare -a FAILED_CASES

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  [단계 1] IC 재현 학습 시작"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    for FULL_ID in "${TEST_PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            TOTAL=$((TOTAL + 1))
            PATIENT="${FULL_ID##*_}"   # "003", "126", "354"

            PATIENT_DIR="${REPRO_ROOT}/${FULL_ID}"
            DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_IC_repro.nii.gz"
            DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_IC_repro.pt"
            INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

            mkdir -p "${PATIENT_DIR}"

            # 이미 완료된 케이스 스킵
            if [ -f "${DST}" ]; then
                echo "  [SKIP] 이미 존재: ${FULL_ID} / ${MODALITY}"
                SUCCESS=$((SUCCESS + 1))
                continue
            fi

            # 입력 파일 확인
            if [ ! -f "${INPUT}" ]; then
                echo "  ❌ 입력 파일 없음: ${FULL_ID} / ${MODALITY}"
                echo "     경로: ${INPUT}"
                FAILED_CASES+=("${FULL_ID}_${MODALITY}")
                continue
            fi

            echo ""
            echo "  ================================================================"
            echo "  [TRAIN] Patient=${PATIENT}  Modality=${MODALITY}"
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
        echo "  ⚠️  학습 실패 케이스가 있습니다. 평가를 진행하려면 확인 후 재실행하세요."
    else
        echo "  ✅ 모든 케이스 학습 완료"
    fi
else
    echo "  [SKIP] 학습 (SKIP_TRAIN=1)"
fi

# ────────────────────────────────────────────────────────────────
# 단계 2: 임시 evaluate_batch_v2.py 설정 주입 + 평가 실행
# ────────────────────────────────────────────────────────────────
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  [단계 2] 이미지 품질 평가 시작"
    echo "  비교: Ours_orig(recon_IC) vs Repro(recon_IC_repro)"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    mkdir -p "${EVAL_DIR}"

    # 3명 환자 ID만 담은 임시 test.csv 생성
    TEMP_CSV="${EVAL_DIR}/_repro_test_patients.csv"
    echo "Brats20ID" > "${TEMP_CSV}"
    for FULL_ID in "${TEST_PATIENTS[@]}"; do
        echo "${FULL_ID}" >> "${TEMP_CSV}"
    done
    echo "  [INFO] 임시 평가용 CSV 생성: ${TEMP_CSV}"

    # evaluate_batch_v2.py 를 인-플레이스 설정 오버라이드 없이 실행하기 위해
    # 임시 평가 스크립트를 생성한다.
    # (원본 evaluate_batch_v2.py를 건드리지 않음)
    TEMP_EVAL="${EVAL_DIR}/_ic_repro_eval.py"
    cat > "${TEMP_EVAL}" << 'PYEOF'
"""
_ic_repro_eval.py  —  evaluate_batch_v2.py 설정 오버라이드 래퍼
IC 재현 실험: Ours_orig (recon_IC) vs Repro (recon_IC_repro) 비교용.
"""
import importlib.util, sys, os

# ── 설정 오버라이드 ────────────────────────────────────────────
# 환경변수로 주입 (run_ic_repro_test.sh 가 설정)
NESVOR_ROOT = os.environ["NESVOR_ROOT"]
RESULT_CSV  = os.environ["RESULT_CSV"]
TEMP_CSV    = os.environ["TEMP_CSV"]
REPRO_ROOT  = os.environ["REPRO_ROOT"]
ORIG_ROOT   = os.environ["ORIG_ROOT"]

# ── evaluate_batch_v2 모듈 동적 로드 ──────────────────────────
spec = importlib.util.spec_from_file_location(
    "evaluate_batch_v2",
    os.path.join(NESVOR_ROOT, "evaluate_batch_v2.py"),
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# ── 설정 덮어쓰기 ──────────────────────────────────────────────
m.TEST_CSV   = TEMP_CSV
m.RESULT_CSV = RESULT_CSV
m.METHODS    = {
    "Ours_orig" : ORIG_ROOT,   # 원본 최고성능 결과 (기준)
    "Repro"     : REPRO_ROOT,  # IC 브랜치 재현 결과 (비교 대상)
}

# ── 실행 ───────────────────────────────────────────────────────
m.main()
PYEOF

    echo "  [INFO] 임시 평가 스크립트 생성: ${TEMP_EVAL}"

    # nerf_tcnn conda 환경에서 실행
    export NESVOR_ROOT RESULT_CSV TEMP_CSV REPRO_ROOT ORIG_ROOT
    conda run \
        -n /dshome/ddualab/conda_env/nerf_tcnn \
        --no-capture-output \
        python "${TEMP_EVAL}"

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → ${RESULT_CSV}"
    fi

    # 임시 파일 정리 (평가 성공 시에만)
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
echo "  IC 재현 실험 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
fi
echo "  재현 결과 경로: ${REPRO_ROOT}"
echo "  평가 결과 CSV : ${RESULT_CSV}"
echo "================================================================"

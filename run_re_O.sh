#!/usr/bin/env bash
# ================================================================
# run_re_O.sh  (재현 스크립트 — re_O 브랜치)
# hsm_O (alpha=0.50, start_iter=0) 단독 재현 실행
#
# 기준 커밋 : 3e7858f  (2026-04-30)
# 브랜치    : re_O
# 원본 세트 : run_hsm_dataset.sh 의 hsm_O 블록
#
# 목적:
#   - hsm_O 결과(26.0 dB 전후)를 동일 코드·동일 인자로 재현
#   - 코드 변화 없이 수치가 달라지는지 확인
#   - 재현 완료 후 IC 기준 PSNR/SSIM/LPIPS 비교 평가
#
# 사용법:
#   chmod +x run_re_O.sh
#   ./run_re_O.sh                # 전체 실행 + 평가
#   SKIP_EVAL=1 ./run_re_O.sh   # 평가 건너뜀
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_BASE="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="${RECON_BASE}/test.csv"

TAG="hsm_O"
ALPHA="0.5"
RECON_ROOT="${RECON_BASE}/recon_re_O"   # ← 원본 hsm_O 폴더와 분리

MODALITIES=(flair t1ce)
INTENSITY_MEAN=308

# ── BASE_ARGS (run_hsm_dataset.sh 와 완전 동일) ───────────────
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
"

FAILED_CASES=()
TOTAL=0
SUCCESS=0

mkdir -p "${RECON_ROOT}"

echo ""
echo "################################################################"
echo "  재현 실행: ${TAG}  (alpha=${ALPHA}, start_iter=0)"
echo "  출력 루트: ${RECON_ROOT}"
echo "  기준 커밋: 3e7858f  (re_O 브랜치)"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

# ── test.csv 기반 순차 학습 ───────────────────────────────────
mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

for FULL_ID in "${FULL_IDS[@]}"; do
    PATIENT="${FULL_ID##*_}"
    PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
    mkdir -p "${PATIENT_DIR}"

    for MODALITY in "${MODALITIES[@]}"; do
        DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${TAG}.nii.gz"

        if [ -f "${DST}" ]; then
            echo "    [SKIP] 이미 존재: ${PATIENT} / ${MODALITY}"
            continue
        fi

        INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

        if [ ! -f "${INPUT}" ]; then
            echo "    ❌ 입력 없음: ${PATIENT} / ${MODALITY}  (${INPUT})"
            FAILED_CASES+=("${PATIENT}_${MODALITY}_input_missing")
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
            --slice-residual-alpha "${ALPHA}" \
            --hard-mining-start-iter 0

        EXIT_CODE=$?
        TOTAL=$((TOTAL + 1))
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "  ❌ FAILED: ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
            FAILED_CASES+=("${PATIENT}_${MODALITY}")
        else
            SUCCESS=$((SUCCESS + 1))
            echo "  ✅ DONE : ${PATIENT} / ${MODALITY}"
            echo "       → ${DST}"
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
        fi
    done
done

# ── 실행 요약 ─────────────────────────────────────────────────
echo ""
echo "  ── 재현 실행 완료: $(date '+%Y-%m-%d %H:%M:%S') ──"
echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    echo "  ❌ 실패 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "       - ${c}"
    done
else
    echo "  ✅ 모든 케이스 성공"
fi

# ── 평가 ─────────────────────────────────────────────────────────
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  평가 시작: eval_hsm_dataset.py"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    conda run -n nerf_tcnn python eval_hsm_dataset.py

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_results/eval_hsm_dataset.csv"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

echo ""
echo "================================================================"
echo "  전체 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

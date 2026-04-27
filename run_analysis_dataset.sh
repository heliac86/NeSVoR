#!/usr/bin/env bash
# ================================================================
# run_analysis_dataset.sh  (v1 — 2026-04-27)
# Hard Slice Mining 분석용 재학습 스크립트
# IC 확정 설정으로 test셋 48명 × 2 모달리티(flair, t1ce) = 96 cases
#
# ── 목적 ──────────────────────────────────────────────────────
#   각 케이스 학습 종료 후 slice_analysis/ 디렉토리에 아래 파일 저장:
#     - slice_residuals_final.npy       : 최종 슬라이스 잔차 EMA
#     - slice_sample_counts_main.npy    : 메인 배치 샘플링 카운트
#     - slice_sample_counts_patch.npy   : 패치 샘플링 카운트
#     - slice_pixel_counts.npy          : 슬라이스별 픽셀 수
#     - residuals_history_iters.npy     : 스냅샷 iter 번호
#     - residuals_history_values.npy    : iter별 잔차 EMA 시계열
#
# ── 출력 구조 ─────────────────────────────────────────────────
#   recon_analysis/
#     BraTS20_Training_003/
#       003_flair_4x5_analysis.nii.gz
#       003_flair_4x5_analysis.pt
#       003_flair_4x5_analysis/
#         slice_analysis/
#           slice_residuals_final.npy
#           ...
#       003_t1ce_4x5_analysis.nii.gz
#       ...
#
# ── 입력 경로 ─────────────────────────────────────────────────
#   test.csv      : /dshome/ddualab/dongnyeok/NeSVoR/test.csv
#   degraded 원본 : /data/BraTS20_Degraded_4x_5/
#
# 사용법:
#   chmod +x run_analysis_dataset.sh
#   ./run_analysis_dataset.sh
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_analysis"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# ── 모달리티 ─────────────────────────────────────────────────
MODALITIES=(flair t1ce)

# ── IC 확정 학습 설정 ─────────────────────────────────────────
INTENSITY_MEAN=308

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

echo ""
echo "################################################################"
echo "  Hard Slice Mining 분석용 재학습 시작"
echo "  CSV  : ${TEST_CSV}"
echo "  OUT  : ${RECON_ROOT}"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

# csv에서 Brats20ID 칼럼 읽기 (헤더 제외)
mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

for FULL_ID in "${FULL_IDS[@]}"; do
    # BraTS20_Training_003 → PATIENT=003
    PATIENT="${FULL_ID##*_}"
    PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
    mkdir -p "${PATIENT_DIR}"

    for MODALITY in "${MODALITIES[@]}"; do
        # ── 출력 파일 경로 ──────────────────────────────────────
        OUT_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_analysis.nii.gz"
        OUT_MODEL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_analysis.pt"
        # slice_analysis/ 는 --output-volume 의 부모 디렉토리(PATIENT_DIR) 아래에 생성됨
        # train.py의 ANALYSIS 블록: os.path.dirname(args.output_volume) / slice_analysis/

        # 이미 완료된 케이스 스킵 (볼륨 + 모델 + slice_analysis 모두 존재 시)
        ANALYSIS_DIR="${PATIENT_DIR}/slice_analysis"
        if [ -f "${OUT_VOL}" ] && [ -f "${OUT_MODEL}" ] && [ -d "${ANALYSIS_DIR}" ]; then
            echo ""
            echo "  [SKIP] 이미 완료: ${PATIENT} / ${MODALITY}"
            echo "         ${OUT_VOL}"
            continue
        fi

        INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

        # 입력 파일 존재 확인
        if [ ! -f "${INPUT}" ]; then
            echo ""
            echo "  ❌ 입력 파일 없음: ${PATIENT} / ${MODALITY}"
            echo "     찾는 경로: ${INPUT}"
            FAILED_CASES+=("${PATIENT}_${MODALITY}_input_missing")
            TOTAL=$((TOTAL + 1))
            continue
        fi

        echo ""
        echo "================================================================"
        echo "  [START] Patient=${PATIENT}  Modality=${MODALITY}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"

        # shellcheck disable=SC2086
        nesvor reconstruct \
            --input-stacks   "${INPUT}" \
            --output-volume  "${OUT_VOL}" \
            --output-model   "${OUT_MODEL}" \
            --sample-mask    "${INPUT}" \
            --output-intensity-mean ${INTENSITY_MEAN} \
            ${BASE_ARGS}

        EXIT_CODE=$?
        TOTAL=$((TOTAL + 1))
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "  ❌ FAILED: ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
            FAILED_CASES+=("${PATIENT}_${MODALITY}")
        else
            SUCCESS=$((SUCCESS + 1))
            echo "  ✅ DONE : ${PATIENT} / ${MODALITY}"
            echo "     Vol   → ${OUT_VOL}"
            echo "     Model → ${OUT_MODEL}"
            echo "     Analysis → ${ANALYSIS_DIR}"
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
        fi
    done
done

# ── 최종 요약 ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
echo "  출력 루트: ${RECON_ROOT}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패한 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo "================================================================"

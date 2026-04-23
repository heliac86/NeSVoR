#!/usr/bin/env bash
# ================================================================
# run_gating_analysis.sh  (v1 — 2026-04-23)
# 게이팅 가중치 분석용: 48명 × 4 모달리티 재구성 + 모델 체크포인트 저장
#
# ── 목적 ──────────────────────────────────────────────────────
#   각 볼륨 학습 후 model.pt를 저장하여,
#   INR.level_weights 수렴값을 모달리티별로 분석하기 위한 데이터 수집
#
# ── 확정 설정 (time_IC 기준) ──────────────────────────────────
#   weight-ff-loss=0.10, ff-alpha=1.0
#   weight-diversity-loss=0.08, target-diversity-var=0.06
#   hard-mining-main-loss
#
# ── 출력 구조 ─────────────────────────────────────────────────
#   recon_gating/
#     BraTS20_Training_003/
#       003_flair_gating.nii.gz
#       003_flair_gating_model.pt    ← level_weights 분석용
#       003_t1ce_gating.nii.gz
#       003_t1ce_gating_model.pt
#       003_t2_gating.nii.gz
#       003_t2_gating_model.pt
#       003_t1_gating.nii.gz
#       003_t1_gating_model.pt
#     ...
#
# 사용법:
#   chmod +x run_gating_analysis.sh
#   ./run_gating_analysis.sh
#   SKIP_TRAIN=1 ./run_gating_analysis.sh   # 학습 건너뜀 (디버그용)
#
# 환경변수로 특정 모달리티만 실행:
#   MODALITY_FILTER=flair ./run_gating_analysis.sh
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_gating"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# ── 플래그 ────────────────────────────────────────────────────
SKIP_TRAIN=${SKIP_TRAIN:-0}

# ── 모달리티 설정 ─────────────────────────────────────────────
# 환경변수 MODALITY_FILTER 로 특정 모달리티만 실행 가능
# 예: MODALITY_FILTER=t2 ./run_gating_analysis.sh
ALL_MODALITIES=(flair t1ce t2 t1)
if [ -n "${MODALITY_FILTER}" ]; then
    MODALITIES=("${MODALITY_FILTER}")
    echo "  [FILTER] 모달리티 필터 적용: ${MODALITY_FILTER}"
else
    MODALITIES=("${ALL_MODALITIES[@]}")
fi

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
  --weight-ff-loss 0.10
  --ff-alpha 1.0
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

# ================================================================
# 메인 학습 루프: test.csv 기반 48명 × 4 모달리티
# ================================================================
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  게이팅 가중치 분석용 재구성 시작"
    echo "  대상: 48명 × ${#MODALITIES[@]} 모달리티 = $((48 * ${#MODALITIES[@]})) 케이스"
    echo "  CSV : ${TEST_CSV}"
    echo "  출력: ${RECON_ROOT}"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    # csv에서 Brats20ID 칼럼 읽기 (헤더 제외)
    mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

    for FULL_ID in "${FULL_IDS[@]}"; do
        PATIENT="${FULL_ID##*_}"
        PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
        mkdir -p "${PATIENT_DIR}"

        for MODALITY in "${MODALITIES[@]}"; do
            DST_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_gating.nii.gz"
            DST_MODEL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_gating_model.pt"

            # 볼륨 + 모델 둘 다 있으면 스킵
            if [ -f "${DST_VOL}" ] && [ -f "${DST_MODEL}" ]; then
                echo ""
                echo "  [SKIP] 이미 완료: ${PATIENT} / ${MODALITY}"
                continue
            fi

            INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

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
            echo "  Input : ${INPUT}"
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            echo "================================================================"

            # shellcheck disable=SC2086
            nesvor reconstruct \
                --input-stacks   "${INPUT}" \
                --output-volume  "${DST_VOL}" \
                --output-model   "${DST_MODEL}" \
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
                echo "  Vol   → ${DST_VOL}"
                echo "  Model → ${DST_MODEL}"
                echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            fi
        done
    done

else
    echo "  [SKIP] 학습 전체 건너뜀 (SKIP_TRAIN=1)"
fi

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

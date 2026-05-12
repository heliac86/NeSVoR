#!/usr/bin/env bash
# ================================================================
# run_gating_hsm_O.sh  (2026-05-12)
# 게이팅 가중치 분석용: 48명 × 4 모달리티 재구성 + 모델 체크포인트 저장
#
# ── 목적 ──────────────────────────────────────────────────────────
#   hsm_O 설정과 no_gating 설정 각각으로 학습 후 model.pt를 저장하여
#   analyze_gating_weights.py 에 입력할 level_weights 데이터 수집
#
# ── 설정 기준 ─────────────────────────────────────────────────────
#   run_ablation_O.sh 의 BASE_ARGS_COMMON + 조건별 인자를 그대로 사용
#
# ── 출력 구조 ─────────────────────────────────────────────────────
#   recon_gating_hsm_O/
#     hsm_O/
#       BraTS20_Training_003/
#         003_flair_gating_model.pt    ← analyze_gating_weights.py 입력
#         003_flair_gating.nii.gz
#         003_t1ce_gating_model.pt
#         003_t1ce_gating.nii.gz
#         003_t2_gating_model.pt
#         ...
#     no_gating/
#       BraTS20_Training_003/
#         003_flair_gating_model.pt
#         ...
#
# ── 사용법 ───────────────────────────────────────────────────────
#   chmod +x run_gating_hsm_O.sh
#   ./run_gating_hsm_O.sh                  # 두 조건 모두 실행
#   COND_FILTER=hsm_O ./run_gating_hsm_O.sh      # hsm_O 만
#   COND_FILTER=no_gating ./run_gating_hsm_O.sh  # no_gating 만
#   MODALITY_FILTER=flair ./run_gating_hsm_O.sh  # 특정 모달리티만
#   SKIP_TRAIN=1 ./run_gating_hsm_O.sh           # 학습 건너뜀 (디버그)
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 경로 설정 ─────────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_gating_hsm_O"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

INTENSITY_MEAN=308
SKIP_TRAIN=${SKIP_TRAIN:-0}

# ── 모달리티 설정 ─────────────────────────────────────────────────
ALL_MODALITIES=(flair t1ce t2 t1)
if [ -n "${MODALITY_FILTER}" ]; then
    MODALITIES=("${MODALITY_FILTER}")
    echo "  [FILTER] 모달리티 필터 적용: ${MODALITY_FILTER}"
else
    MODALITIES=("${ALL_MODALITIES[@]}")
fi

# ── 실행 조건 설정 ────────────────────────────────────────────────
ALL_CONDS=(hsm_O no_gating)
if [ -n "${COND_FILTER}" ]; then
    CONDS=("${COND_FILTER}")
    echo "  [FILTER] 조건 필터 적용: ${COND_FILTER}"
else
    CONDS=("${ALL_CONDS[@]}")
fi

# ── 공통 BASE_ARGS (run_ablation_O.sh 의 BASE_ARGS_COMMON 동일) ───
BASE_ARGS_COMMON="
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
  --hard-mining-start-iter 0
  --slice-residual-alpha 0.5
  --hsm-beta 0.0
"

# ── 조건별 추가 ARGS ──────────────────────────────────────────────
# hsm_O: Ours 기준선 (run_ablation_O.sh 의 A1~A4 공통 기반 설정과 동일)
ARGS_hsm_O="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# no_gating: run_ablation_O.sh 의 ARGS_no_gating 과 동일
ARGS_no_gating="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --no-gating
"

# ── 유틸: 단일 케이스 실행 ───────────────────────────────────────
run_one() {
    local FULL_ID="$1"     # BraTS20_Training_003
    local MODALITY="$2"    # flair | t1ce | t2 | t1
    local COND="$3"        # hsm_O | no_gating
    local EXTRA_ARGS="$4"

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${COND}/${FULL_ID}"
    local DST_VOL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_gating.nii.gz"
    local DST_MODEL="${PATIENT_DIR}/${PATIENT}_${MODALITY}_gating_model.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    # 볼륨 + 모델 둘 다 있으면 스킵
    if [ -f "${DST_VOL}" ] && [ -f "${DST_MODEL}" ]; then
        echo "  [SKIP] 이미 완료: ${COND} / ${PATIENT} / ${MODALITY}"
        return 0
    fi

    if [ ! -f "${INPUT}" ]; then
        echo "  ❌ 입력 파일 없음: ${COND} / ${PATIENT} / ${MODALITY}"
        echo "     찾는 경로: ${INPUT}"
        return 1
    fi

    echo ""
    echo "================================================================"
    echo "  [START] Cond=${COND}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks   "${INPUT}" \
        --output-volume  "${DST_VOL}" \
        --output-model   "${DST_MODEL}" \
        --sample-mask    "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    return $?
}

# ── 메인 루프 ────────────────────────────────────────────────────
TOTAL=0
SUCCESS=0
FAILED_CASES=()

if [ "${SKIP_TRAIN}" -eq 0 ]; then
    mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

    for COND in "${CONDS[@]}"; do
        case "${COND}" in
            hsm_O)     EXTRA_ARGS="${ARGS_hsm_O}" ;;
            no_gating) EXTRA_ARGS="${ARGS_no_gating}" ;;
            *)
                echo "  ❌ 알 수 없는 조건: ${COND}  (hsm_O | no_gating)"
                exit 1
                ;;
        esac

        echo ""
        echo "################################################################"
        echo "  조건: ${COND}"
        echo "  대상: 48명 × ${#MODALITIES[@]} 모달리티"
        echo "  출력: ${RECON_ROOT}/${COND}"
        echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "################################################################"

        for FULL_ID in "${FULL_IDS[@]}"; do
            for MODALITY in "${MODALITIES[@]}"; do
                TOTAL=$((TOTAL + 1))
                if run_one "${FULL_ID}" "${MODALITY}" "${COND}" "${EXTRA_ARGS}"; then
                    SUCCESS=$((SUCCESS + 1))
                else
                    FAILED_CASES+=("${COND}_${FULL_ID##*_}_${MODALITY}")
                fi
            done
        done

        echo ""
        echo "  [완료] 조건 ${COND}: $(date '+%Y-%m-%d %H:%M:%S')"
    done

else
    echo "  [SKIP] 학습 전체 건너뜀 (SKIP_TRAIN=1)"
fi

# ── 최종 요약 ────────────────────────────────────────────────────
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

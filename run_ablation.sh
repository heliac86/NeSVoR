#!/usr/bin/env bash
# ================================================================
# run_ablation.sh  (v1 — 2026-04-24)
# Ablation Study: 세 가지 모듈을 하나씩 제거한 변형 실험
#
# ── Ablation 조건 ─────────────────────────────────────────────
#   A1: w/o FF Loss      → --weight-ff-loss 0.0
#   A2: w/o Hard Mining  → --hard-mining-main-loss 제거
#   A3: w/o Gating       → --no-gating (Diversity Loss도 자동 비활성화)
#
# ── 공통 베이스 설정 (run_ic_dataset.sh 와 동일) ───────────────
#   n_iter=2000, batch=4096, lr=5e-3, gamma=0.33
#   milestones=[0.5, 0.75, 0.9]
#   weight-ff-loss=0.05, ff-alpha=2.0
#   weight-diversity-loss=0.08, hard-mining-main-loss
#
# ── 처리 대상 ────────────────────────────────────────────────
#   test.csv의 48명 환자, flair + t1ce 두 모달리티
#
# ── 출력 구조 ─────────────────────────────────────────────────
#   recon_ablation/
#     no_ff/
#       BraTS20_Training_003/
#         003_flair_4x5_no_ff.nii.gz
#         003_t1ce_4x5_no_ff.nii.gz
#     no_hm/
#       BraTS20_Training_003/
#         003_flair_4x5_no_hm.nii.gz
#         ...
#     no_gating/
#       BraTS20_Training_003/
#         003_flair_4x5_no_gating.nii.gz
#         ...
#
# 사용법:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh                         # A1 + A2 + A3 전체 실행
#   ABLATION=no_ff ./run_ablation.sh          # A1만 실행
#   ABLATION=no_hm ./run_ablation.sh          # A2만 실행
#   ABLATION=no_gating ./run_ablation.sh      # A3만 실행
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_ablation"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── 어떤 Ablation을 실행할지 결정 ─────────────────────────────
# 환경변수 ABLATION이 설정되어 있으면 해당 조건만 실행
# 설정되어 있지 않으면 세 조건 모두 실행
if [ -n "${ABLATION}" ]; then
    ABLATION_TARGETS=("${ABLATION}")
else
    ABLATION_TARGETS=(no_ff no_hm no_gating)
fi

# ── 공통 BASE ARGS (Full IC 설정, 모듈 제거 전 기준) ────────────
BASE_ARGS_COMMON="
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
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
"

# ── 조건별 추가/제거 ARGS ──────────────────────────────────────
# A1: w/o FF Loss — weight-ff-loss를 0.0으로 설정 (ff-alpha도 무관하나 명시)
ARGS_no_ff="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.0
  --ff-alpha 2.0
"

# A2: w/o Hard Slice Mining — --hard-mining-main-loss 플래그 제거
ARGS_no_hm="${BASE_ARGS_COMMON}
  --weight-ff-loss 0.05
  --ff-alpha 2.0
"

# A3: w/o Gating — --no-gating 추가 (Diversity Loss는 자동 비활성화)
ARGS_no_gating="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --no-gating
"

# ── 유틸: 단일 (환자, 모달리티, 조건) 실행 ──────────────────────
run_one() {
    local FULL_ID="$1"    # BraTS20_Training_003
    local MODALITY="$2"   # flair | t1ce
    local COND="$3"       # no_ff | no_hm | no_gating
    local EXTRA_ARGS="$4" # 조건별 args 문자열

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${COND}/${FULL_ID}"
    local DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${COND}.nii.gz"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    # 이미 완료된 케이스 스킵
    if [ -f "${DST}" ]; then
        echo "  [SKIP] 이미 존재: ${COND} / ${PATIENT} / ${MODALITY}"
        return 0
    fi

    # 입력 파일 존재 확인
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
        --input-stacks  "${INPUT}" \
        --output-volume "${DST}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    return $?
}

# ── 메인 루프 ─────────────────────────────────────────────────
TOTAL=0
SUCCESS=0
declare -a FAILED_CASES

mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

for COND in "${ABLATION_TARGETS[@]}"; do

    # 조건별 ARGS 선택
    case "${COND}" in
        no_ff)     EXTRA_ARGS="${ARGS_no_ff}" ;;
        no_hm)     EXTRA_ARGS="${ARGS_no_hm}" ;;
        no_gating) EXTRA_ARGS="${ARGS_no_gating}" ;;
        *)
            echo "  ❌ 알 수 없는 Ablation 조건: ${COND}"
            echo "     유효한 값: no_ff | no_hm | no_gating"
            exit 1
            ;;
    esac

    echo ""
    echo "################################################################"
    echo "  Ablation 조건: ${COND}"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
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

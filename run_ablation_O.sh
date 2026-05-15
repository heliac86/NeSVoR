#!/usr/bin/env bash
# ================================================================
# run_ablation_O.sh  (v2 — 2026-05-15)
# Ablation Study (hsm_O 기준선)
#
# ── 기준선 변경 이유 ──────────────────────────────────────────────
#   이전 run_ablation.sh 는 time_IC 버전 기반이었음.
#   현재 최고 성능 세팅은 hsm_O (alpha=0.5, beta=0.0)이므로
#   어블레이션 기준을 hsm_O 로 갱신함.
#
# ── Ablation 조건 ─────────────────────────────────────────────────
#   A0: Ours_O           → 전체 기능 ON (기준선 그 자체)
#   A1: w/o FF Loss      → --weight-ff-loss 0.0
#   A2: w/o Hard Mining  → --no-hard-mining  (main + patch 양쪽 HSM 제거)
#   A3: w/o Gating       → --no-gating       (level_weights + diversity loss 제거)
#   A4: w/o Diversity    → --weight-diversity-loss 0.0
#                          (level_weights 유지, diversity loss만 제거)
#                          gating 전체 vs diversity loss 기여 분리 목적
#
# ── 공통 BASE ARGS (hsm_O 기준 — run_optionC_sweep.sh 의 BASE_ARGS 기반) ────
#   alpha=0.5  (hsm_O 동작)
#   hard-mining-main-loss 포함
#   weight-ff-loss=0.05, ff-alpha=2.0
#   weight-diversity-loss=0.08, target-diversity-var=0.06
#   diversity-loss-fn=variance, diversity-loss-space=raw
#   gating-grad-clip=0.0
#   n-iter=2000, batch=4096, lr=5e-3
#   gamma=0.33, milestones=0.5 0.75 0.9, seed=42
#   hard-mining-start-iter=0
#   ※ --hsm-beta 는 hsm_O 브랜치에 없으므로 사용하지 않음
#
# ── 처리 대상 ─────────────────────────────────────────────────────
#   test.csv 의 48명 환자, flair + t1ce 두 모달리티
#
# ── 출력 구조 ─────────────────────────────────────────────────────
#   recon_ablation_O/
#     Ours_O/
#       BraTS20_Training_003/
#         003_flair_4x5_Ours_O.nii.gz
#         003_flair_4x5_Ours_O.pt
#         003_t1ce_4x5_Ours_O.nii.gz
#         003_t1ce_4x5_Ours_O.pt
#     no_ff/
#       BraTS20_Training_003/
#         003_flair_4x5_no_ff.nii.gz
#         003_flair_4x5_no_ff.pt
#         ...
#     no_hm/   ...
#     no_gating/   ...
#     no_diversity/   ...
#
# ── 사용법 ───────────────────────────────────────────────────────
#   chmod +x run_ablation_O.sh
#   ./run_ablation_O.sh                           # A0~A4 전체 실행
#   ABLATION=Ours_O       ./run_ablation_O.sh    # A0만
#   ABLATION=no_ff        ./run_ablation_O.sh    # A1만
#   ABLATION=no_hm        ./run_ablation_O.sh    # A2만
#   ABLATION=no_gating    ./run_ablation_O.sh    # A3만
#   ABLATION=no_diversity ./run_ablation_O.sh    # A4만
#   SKIP_EVAL=1           ./run_ablation_O.sh    # 평가 건너뜀
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ─────────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_ablation_O"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

INTENSITY_MEAN=308
MODALITIES=(flair t1ce)

# ── 어떤 Ablation을 실행할지 결정 ────────────────────────────────
# 환경변수 ABLATION이 설정되어 있으면 해당 조건만 실행
# 설정되어 있지 않으면 다섯 조건 모두 실행
if [ -n "${ABLATION}" ]; then
    ABLATION_TARGETS=("${ABLATION}")
else
    ABLATION_TARGETS=(Ours_O no_ff no_hm no_gating no_diversity)
fi

# ── 공통 BASE ARGS (hsm_O 기준) ──────────────────────────────────
# ※ --hsm-beta 는 hsm_O 브랜치 get_batch()에 해당 파라미터가 없으므로 제거
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
"

# ── 조건별 ARGS ──────────────────────────────────────────────────
# A0: Ours_O (기준선 — 전체 기능 ON)
#   모든 구성요소가 활성화된 hsm_O 기준 최고 성능 세팅
ARGS_Ours_O="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# A1: w/o FF Loss
#   FF Loss 제거 → 패치 경로 자체가 비활성화되므로 HSM 패치 샘플링도 자동으로 꺼짐
#   gating / diversity loss 는 기준선과 동일하게 유지
ARGS_no_ff="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.0
  --ff-alpha 2.0
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# A2: w/o Hard Mining
#   --no-hard-mining: main MSE 픽셀 경로 + FF 패치 경로 양쪽 HSM 완전 비활성화
#   (train.py: no_hard_mining=True → sampling_probs=None 강제)
#   gating / diversity loss 는 기준선과 동일하게 유지
ARGS_no_hm="${BASE_ARGS_COMMON}
  --no-hard-mining
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# A3: w/o Gating
#   --no-gating: level_weights 파라미터 자체를 생성하지 않음
#   → hash grid 모든 레벨이 raw 균등 합산 (가중치 없음)
#   → diversity loss도 자동 비활성화 (train.py 내 use_diversity_loss 조건)
#   HSM / FF Loss 는 기준선과 동일하게 유지
ARGS_no_gating="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --no-gating
"

# A4: w/o Diversity Loss
#   level_weights 파라미터는 그대로 학습됨 (gating 유지)
#   diversity loss만 0으로 설정 → 자연 분화 여부 확인
#   no_gating 과의 비교: level_weights 파라미터 자체의 기여 분리
#   Ours_O 와의 비교:    diversity loss 의 기여 분리
ARGS_no_diversity="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --weight-diversity-loss 0.0
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# ── 유틸: 단일 (환자, 모달리티, 조건) 실행 ──────────────────────
run_one() {
    local FULL_ID="$1"    # BraTS20_Training_003
    local MODALITY="$2"   # flair | t1ce
    local COND="$3"       # Ours_O | no_ff | no_hm | no_gating | no_diversity
    local EXTRA_ARGS="$4" # 조건별 args 문자열

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${COND}/${FULL_ID}"
    local DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${COND}.nii.gz"
    local DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${COND}.pt"
    local INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

    mkdir -p "${PATIENT_DIR}"

    # 이미 완료된 케이스 스킵 (.nii.gz 기준)
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
        --output-model  "${DST_PT}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    return $?
}

# ── 메인 루프 ────────────────────────────────────────────────────
TOTAL=0
SUCCESS=0
declare -a FAILED_CASES

mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

for COND in "${ABLATION_TARGETS[@]}"; do

    case "${COND}" in
        Ours_O)       EXTRA_ARGS="${ARGS_Ours_O}" ;;
        no_ff)        EXTRA_ARGS="${ARGS_no_ff}" ;;
        no_hm)        EXTRA_ARGS="${ARGS_no_hm}" ;;
        no_gating)    EXTRA_ARGS="${ARGS_no_gating}" ;;
        no_diversity) EXTRA_ARGS="${ARGS_no_diversity}" ;;
        *)
            echo "  ❌ 알 수 없는 Ablation 조건: ${COND}"
            echo "     유효한 값: Ours_O | no_ff | no_hm | no_gating | no_diversity"
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
        echo "       - ${c}"
    done
fi
echo "================================================================"

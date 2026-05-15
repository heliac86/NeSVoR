#!/usr/bin/env bash
# ================================================================
# run_ablation_P.sh  (v1 — 2026-05-15)
# Ablation Study (exp_P 기준선)
#
# ── 기준선 변경 이유 ──────────────────────────────────────────────
#   exp_P: hsm_O 코드 기반에서 ff_direct를 gating_mode 기본값으로
#   변경한 브랜치. no_gating 시 ff_loss가 encoding 업데이트를
#   차단(encoding.detach)하여 density_net만 업데이트함.
#
# ── Ablation 조건 ─────────────────────────────────────────────────
#   Ours_P:      기준선 전체 (ff_direct 기본값 + gating + diversity)
#   no_ff:       w/o FF Loss      → --weight-ff-loss 0.0
#   no_hm:       w/o Hard Mining  → --no-hard-mining
#   no_gating:   w/o Gating       → --no-gating
#                (gating_mode=ff_direct 기본값 유지 → encoding detach)
#   no_diversity:w/o Diversity    → --weight-diversity-loss 0.0
#
# ── 공통 BASE ARGS (exp_P 기준) ──────────────────────────────────
#   --hsm-beta 미사용 (exp_P/hsm_O 에서 제거됨)
#   alpha=0.5, hard-mining-main-loss 포함
#   weight-ff-loss=0.05, ff-alpha=2.0
#   weight-diversity-loss=0.08, target-diversity-var=0.06
#   diversity-loss-fn=variance, diversity-loss-space=raw
#   gating-grad-clip=0.0
#   n-iter=2000, batch=4096, lr=5e-3
#   gamma=0.33, milestones=0.5 0.75 0.9, seed=42
#   hard-mining-start-iter=0
#
# ── 처리 대상 ─────────────────────────────────────────────────────
#   test.csv 의 48명 환자, flair + t1ce 두 모달리티
#
# ── 출력 구조 ─────────────────────────────────────────────────────
#   recon_ablation_P/
#     Ours_P/
#       BraTS20_Training_003/
#         003_flair_4x5_Ours_P.nii.gz
#         003_flair_4x5_Ours_P.pt
#         003_t1ce_4x5_Ours_P.nii.gz
#         003_t1ce_4x5_Ours_P.pt
#     no_ff/   ...
#     no_hm/   ...
#     no_gating/   ...
#     no_diversity/   ...
#
# ── 사용법 ───────────────────────────────────────────────────────
#   chmod +x run_ablation_P.sh
#   ./run_ablation_P.sh                           # 전체 5개 조건 실행
#   ABLATION=Ours_P       ./run_ablation_P.sh    # Ours_P만
#   ABLATION=no_ff        ./run_ablation_P.sh    # no_ff만
#   ABLATION=no_hm        ./run_ablation_P.sh    # no_hm만
#   ABLATION=no_gating    ./run_ablation_P.sh    # no_gating만
#   ABLATION=no_diversity ./run_ablation_P.sh    # no_diversity만
#   SKIP_EVAL=1           ./run_ablation_P.sh    # 평가 건너뜀
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ─────────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR_P/recon_ablation_P"
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
    ABLATION_TARGETS=(Ours_P no_ff no_hm no_gating no_diversity)
fi

# ── 공통 BASE ARGS (exp_P 기준) ──────────────────────────────────
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

# Ours_P: 기준선 전체
#   gating_mode=ff_direct (기본값) + use_gating=True + diversity loss 모두 포함
ARGS_Ours_P="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
"

# no_ff: w/o FF Loss
#   FF Loss 제거 → 패치 경로 비활성화, HSM 패치 샘플링도 자동으로 꺼짐
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

# no_hm: w/o Hard Mining
#   --no-hard-mining: main MSE + FF 패치 경로 양쪽 HSM 완전 비활성화
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

# no_gating: w/o Gating
#   --no-gating: level_weights 파라미터 미생성, diversity loss 자동 비활성화
#   gating_mode=ff_direct (기본값) 유지 → encoding.detach() 적용
#   → ff_loss가 density_net만 업데이트 (encoding 차단)
#   HSM / FF Loss 는 기준선과 동일하게 유지
ARGS_no_gating="${BASE_ARGS_COMMON}
  --hard-mining-main-loss
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --no-gating
"

# no_diversity: w/o Diversity Loss
#   level_weights 파라미터는 그대로 학습됨 (gating 유지)
#   diversity loss만 0으로 설정 → 자연 분화 여부 확인
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
    local COND="$3"       # Ours_P | no_ff | no_hm | no_gating | no_diversity
    local EXTRA_ARGS="$4" # 조건별 args 문자열

    local PATIENT="${FULL_ID##*_}"
    local PATIENT_DIR="${RECON_ROOT}/${COND}/${FULL_ID}"
    local DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${COND}.nii.gz"
    local DST_PT="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${COND}.pt"
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
        Ours_P)       EXTRA_ARGS="${ARGS_Ours_P}" ;;
        no_ff)        EXTRA_ARGS="${ARGS_no_ff}" ;;
        no_hm)        EXTRA_ARGS="${ARGS_no_hm}" ;;
        no_gating)    EXTRA_ARGS="${ARGS_no_gating}" ;;
        no_diversity) EXTRA_ARGS="${ARGS_no_diversity}" ;;
        *)
            echo "  ❌ 알 수 없는 Ablation 조건: ${COND}"
            echo "     유효한 값: Ours_P | no_ff | no_hm | no_gating | no_diversity"
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

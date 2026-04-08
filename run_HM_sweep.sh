#!/usr/bin/env bash
# ================================================================
# run_HM_sweep.sh
# Hard Mining (HM2) + FF Loss 조합 탐색 스윗
#
# 실험 배경:
#   G7B 스윗 결과, 공간 적응형 게이팅(G6/G7)과 entropy/gini
#   diversity loss가 모두 G5_lowvar를 넘지 못함.
#     G5_lowvar : PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414  ← 현재 최고
#     B1_entropy_lo : 25.537 (2위)
#     G7_h32_warm   : 24.126 (이상치, warmup 중 z_repr 불안정 수렴)
#
#   따라서 전역 hash grid gating(G5_lowvar 조건)을 기반으로
#   아직 단독 검증이 이루어지지 않은 두 기법을 탐색한다.
#
#   [HM2] Hard Slice Mining for get_batch()
#     --hard-mining-main-loss 플래그 추가.
#     slice_residuals EMA를 이용해 어려운 슬라이스를 더 자주 샘플링.
#     기존에는 get_patch_batch(FF Loss용)에만 hard mining이 적용됐으나,
#     이번에 메인 MSE 학습에도 확장.
#
#   [FF] Focal Frequency Loss
#     --weight-ff-loss 0.1 고정 (G7B 스윗에서 사용한 값).
#     FF Loss 단독 효과를 G5_lowvar 기반에서 재확인.
#
#   두 기법의 독립 효과와 시너지를 체계적으로 분리 검증.
#
# 핵심 질문:
#   Q1. HM2 단독이 G5_lowvar를 이기는가?          → HM2 vs G5_repro
#   Q2. FF Loss 단독이 G5_lowvar를 이기는가?       → FF_w010 vs G5_repro
#   Q3. HM2+FF 조합이 각 단독보다 나은가?          → HM2_FF vs HM2, FF_w010
#   Q4. warmup 단독이 G5_lowvar를 이기는가?        → warm500 vs G5_repro
#       (G7B의 warmup 이상치는 z_repr 불안정이 원인이므로,
#        전역 gating에서 단독으로 재검증할 필요 있음)
#   Q5. HM2+warmup 조합이 각 단독보다 나은가?      → HM2_warm vs HM2, warm500
#   Q0. 코드 변경(parsers.py 등) 후 G5_lowvar 수치가 유지되는가?
#       → G5_repro vs 기존 25.664
#
# 실험 세트 (총 6세트, 5명 x 3모달리티 = 15케이스/세트):
#   [0] G5_repro      : G5_lowvar 완전 재현 (코드 드리프트 확인)
#   [1] HM2           : G5_lowvar + hard-mining-main-loss
#   [2] FF_w010       : G5_lowvar + weight-ff-loss=0.1
#   [3] HM2_FF        : G5_lowvar + HM2 + FF
#   [4] warm500       : G5_lowvar + gating-warmup-iters=500
#   [5] HM2_warm      : G5_lowvar + HM2 + warmup=500
#
# 비교 기준선 (재실행 없이 기존 결과 재사용):
#   G5_lowvar : PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 6세트 x 15 = 90
#
# 사용법:
#   chmod +x run_HM_sweep.sh
#   ./run_HM_sweep.sh                        # 모든 세트 실행
#   ENABLE_G5_REPRO=0  ./run_HM_sweep.sh     # G5_repro 건너뜀
#   ENABLE_HM2=0       ./run_HM_sweep.sh     # HM2 건너뜀
#   ENABLE_FF_W010=0   ./run_HM_sweep.sh     # FF_w010 건너뜀
#   ENABLE_HM2_FF=0    ./run_HM_sweep.sh     # HM2_FF 건너뜀
#   ENABLE_WARM500=0   ./run_HM_sweep.sh     # warm500 건너뜀
#   ENABLE_HM2_WARM=0  ./run_HM_sweep.sh     # HM2_warm 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_G5_REPRO=${ENABLE_G5_REPRO:-1}
ENABLE_HM2=${ENABLE_HM2:-1}
ENABLE_FF_W010=${ENABLE_FF_W010:-1}
ENABLE_HM2_FF=${ENABLE_HM2_FF:-1}
ENABLE_WARM500=${ENABLE_WARM500:-1}
ENABLE_HM2_WARM=${ENABLE_HM2_WARM:-1}

# ── 공통 고정 인자 ─────────────────────────────────────────────
# G5_lowvar에서 검증된 최적 조건 그대로 상속.
# spatial gating, z-gating, entropy/gini 계열은 G7B에서 패배 확인 → 제외.
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --weight-image 1.0
  --delta 0.05
  --weight-diversity-loss 0.01
  --target-diversity-var 0.05
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 1.0
"

# FF Loss 공통 인자 (FF 포함 세트에서 추가)
FF_ARGS="
  --weight-ff-loss 0.1
  --patch-size 48
  --n-patches 8
  --ff-alpha 1.0
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ──────────────────────────────────────
run_case() {
    local EXP_TAG=$1
    local PATIENT=$2
    local MODALITY=$3
    shift 3
    local EXTRA_ARGS="$*"

    local INPUT="${DEGRADED_ROOT}/BraTS20_Training_${PATIENT}/BraTS20_Training_${PATIENT}_${MODALITY}.nii"
    local OUTPUT_VOL="${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local OUTPUT_MODEL="${PATIENT}_${MODALITY}_${EXP_TAG}_model.pt"

    echo ""
    echo "================================================================"
    echo "  [START] Tag=${EXP_TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --output-model  "${OUTPUT_MODEL}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        echo "  Vol   → ${OUTPUT_VOL}"
        echo "  Model → ${OUTPUT_MODEL}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# 세트 0: G5_repro
#
#   목적: parsers.py 등 코드 변경 이후에도 G5_lowvar 수치
#         (PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414)가
#         재현되는지 확인. 드리프트 발생 시 이후 결과 전체 무효.
#   변경점: 없음 (G5_lowvar 완전 동일 인자)
#   FF Loss: 미포함 (G7B 스윗의 G5_lowvar도 FF 없이 베스트였음)
# ================================================================
if [ "${ENABLE_G5_REPRO}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 0: G5_repro  (코드 변경 후 재현성 확인)"
    echo "  G5_lowvar 완전 재현 — 인자 변경 없음"
    echo "  기준: PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G5_repro" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS}
        done
    done
else
    echo "  [SKIP] G5_repro (ENABLE_G5_REPRO=0)"
fi

# ================================================================
# 세트 1: HM2
#
#   목적: --hard-mining-main-loss 단독 효과 측정.
#         slice_residuals EMA 기반 가중 샘플링이 메인 MSE 학습에
#         도움이 되는지 G5_repro와 직접 비교.
#   변경점: --hard-mining-main-loss 추가
#   기대: 어려운 슬라이스를 더 자주 보임으로써 전체 수렴 개선.
#   주의: warmup 없이 첫 배치부터 hard mining 활성화됨.
# ================================================================
if [ "${ENABLE_HM2}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: HM2  (Hard Slice Mining for get_batch)"
    echo "  --hard-mining-main-loss"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --hard-mining-main-loss
        done
    done
else
    echo "  [SKIP] HM2 (ENABLE_HM2=0)"
fi

# ================================================================
# 세트 2: FF_w010
#
#   목적: Focal Frequency Loss 단독 효과 측정 (G5_lowvar 기반).
#         G7B 스윗에서도 FF Loss를 사용했으나, 거기서는 spatial
#         gating과 같이 사용되어 FF 단독 기여를 분리하지 못했음.
#         이번에 전역 gating(G5 조건) 위에서 FF 단독 효과 확인.
#   변경점: --weight-ff-loss 0.1 + 관련 패치 인자
#   비교:   G5_repro (FF 없음) vs FF_w010 (FF 있음)
# ================================================================
if [ "${ENABLE_FF_W010}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: FF_w010  (Focal Frequency Loss 단독)"
    echo "  --weight-ff-loss 0.1  --patch-size 48  --n-patches 8"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "FF_w010" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                ${FF_ARGS}
        done
    done
else
    echo "  [SKIP] FF_w010 (ENABLE_FF_W010=0)"
fi

# ================================================================
# 세트 3: HM2_FF
#
#   목적: HM2 + FF Loss 조합의 시너지 검증.
#         HM2(세트 1)와 FF_w010(세트 2)의 단독 효과를 안 뒤에
#         조합이 그 합보다 크거나 같은지 확인.
#         두 기법은 서로 다른 경로로 작동:
#           HM2  → 어떤 슬라이스를 볼지 (샘플링 전략)
#           FF   → 어떤 주파수를 강조할지 (손실 함수)
#         따라서 충돌보다 보완 관계일 가능성이 높음.
#   변경점: --hard-mining-main-loss + FF 인자
# ================================================================
if [ "${ENABLE_HM2_FF}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: HM2_FF  (Hard Mining + FF Loss 조합)"
    echo "  --hard-mining-main-loss"
    echo "  --weight-ff-loss 0.1  --patch-size 48  --n-patches 8"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2_FF" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                ${FF_ARGS} \
                --hard-mining-main-loss
        done
    done
else
    echo "  [SKIP] HM2_FF (ENABLE_HM2_FF=0)"
fi

# ================================================================
# 세트 4: warm500
#
#   목적: gating warmup 단독 효과 측정 (전역 gating 환경에서).
#         G7B에서 G7_h32_warm이 이상치(24.126)를 냈는데, 그것은
#         z_repr(불안정한 density_net 피처)와 warmup 조합의 문제였음.
#         전역 gating(level_weights)에서는 warmup 중 동결 대상이
#         density_net이고 level_weights는 그대로 학습되므로,
#         안정적인 warmup 효과를 기대할 수 있음.
#   변경점: --gating-warmup-iters 500
#   비교:   G5_repro (warmup 없음) vs warm500
# ================================================================
if [ "${ENABLE_WARM500}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 4: warm500  (density_net warmup freeze 500 iter)"
    echo "  --gating-warmup-iters 500"
    echo "  ※ G7_h32_warm과 달리 z_repr 없음 → 이상치 재발 가능성 낮음"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "warm500" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] warm500 (ENABLE_WARM500=0)"
fi

# ================================================================
# 세트 5: HM2_warm
#
#   목적: HM2 + warmup 조합 검증.
#         warmup 초기(500 iter)에 density_net이 동결된 상태에서는
#         slice_residuals의 초기화 품질이 떨어질 수 있음.
#         반면 warmup 이후 density_net이 안정화된 뒤 hard mining이
#         활성화되면 더 신뢰할 수 있는 잔차를 쓰게 됨.
#         → HM2 단독보다 낫다면 "warmup → hard mining" 순서가 효과적.
#   변경점: --hard-mining-main-loss + --gating-warmup-iters 500
#   비교:   HM2(세트 1), warm500(세트 4) vs HM2_warm(세트 5)
# ================================================================
if [ "${ENABLE_HM2_WARM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 5: HM2_warm  (Hard Mining + warmup 500 조합)"
    echo "  --hard-mining-main-loss"
    echo "  --gating-warmup-iters 500"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2_warm" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --hard-mining-main-loss \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] HM2_warm (ENABLE_HM2_WARM=0)"
fi

# ================================================================
# 평가 실행
# G5_lowvar 기존 결과는 재사용 (--exp-tags에 포함하여 비교)
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags G5_lowvar \
               G5_repro \
               HM2 \
               FF_w010 \
               HM2_FF \
               warm500 \
               HM2_warm \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_HM_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_HM_sweep.csv"
fi

# ── 최종 요약 ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  성공: ${SUCCESS} / ${TOTAL}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패한 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo "================================================================"

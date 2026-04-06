#!/usr/bin/env bash
# ================================================================
# run_G7B_sweep.sh
# Gating G7 + 방향B 실험 스윗
#
# 대실험 배경:
#   G5_lowvar(w=0.05, tv=0.12, clip=0.0)이 현재 베스트.
#   G6에서 공간 적응형 GatingMLP의 가능성을 확인한 데 이어,
#   두 가지 독립적인 개선 방향을 병행 검증한다.
#
#   [방향 A — G7] z_repr 조건부 게이팅
#     GatingMLP 입력을 좌표(3D)만 쓰던 G6에서,
#     density_net 중간 피처 z_repr을 추가로 제공하여
#     "해당 위치에 뭐가 있는지"까지 보고 레벨을 선택하게 함.
#     z_repr은 Pass1(ungated forward)에서 detach()로 추출.
#
#     핵심 질문:
#     Q1. z_repr 추가(G7_h32) 자체가 G6_h32보다 나은가?
#         → G7_h32 vs G6_h32 (G6 결과 재사용)
#     Q2. warmup이 z_repr 기반 gating 학습을 돕는가?
#         → G7_h32_warm vs G7_h32
#     Q3. hidden_dim=64로 표현력을 높이면 z_repr 활용도 늘어나는가?
#         → G7_h64 vs G7_h32
#
#   [방향 B — Entropy/Gini Diversity Loss 부호 수정]
#     G5_lowvar의 variance 기반 diversity loss를
#     softmax 공간의 entropy/gini 지표로 대체.
#     entropy/gini는 균등 분포일수록 높으므로,
#     이를 낮추는 방향으로 loss를 걸어 레벨 분화를 유도.
#     (기존 코드의 부호 오류를 수정한 버전.)
#
#     핵심 질문:
#     Q4. entropy 기반 loss(낮은 weight)가 variance 대비 성능 차이?
#         → B1_entropy_lo vs G5_lowvar
#     Q5. entropy weight을 높이면 오히려 불안정해지는가?
#         → B1_entropy_mid vs B1_entropy_lo
#     Q6. gini 기반 loss가 entropy와 비교해 어떤가?
#         → B2_gini_lo vs B1_entropy_lo
#
# 실험 세트 (총 6세트, 5명 x 3모달리티 = 15케이스/세트):
#   [A] G7_h32        spatial-gating + z-gating + h32 + warmup=0
#   [A] G7_h32_warm   spatial-gating + z-gating + h32 + warmup=500
#   [A] G7_h64        spatial-gating + z-gating + h64 + warmup=0
#   [B] B1_entropy_lo  global-gating + entropy + w=0.01
#   [B] B1_entropy_mid global-gating + entropy + w=0.05
#   [B] B2_gini_lo     global-gating + gini    + w=0.01
#
# 비교 기준선 (재실행 불필요):
#   G5_lowvar: variance + w=0.05, raw space — 현재 베스트
#   G6_h32   : spatial-gating + h32, z-gating 없음 (G7과 대조)
#
# 대상 환자: 003 026 030 040 060
# 모달리티 : flair / t2 / t1ce
# 총 케이스: 6세트 x 5명 x 3모달리티 = 90
#
# 사용법:
#   chmod +x run_G7B_sweep.sh
#   ./run_G7B_sweep.sh                            # 모든 세트 실행
#   ENABLE_G7_H32=0      ./run_G7B_sweep.sh       # G7_h32 건너뜀
#   ENABLE_G7_H32_WARM=0 ./run_G7B_sweep.sh       # G7_h32_warm 건너뜀
#   ENABLE_G7_H64=0      ./run_G7B_sweep.sh       # G7_h64 건너뜀
#   ENABLE_B1_ENTROPY_LO=0  ./run_G7B_sweep.sh    # B1_entropy_lo 건너뜀
#   ENABLE_B1_ENTROPY_MID=0 ./run_G7B_sweep.sh    # B1_entropy_mid 건너뜀
#   ENABLE_B2_GINI_LO=0     ./run_G7B_sweep.sh    # B2_gini_lo 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_G7_H32=${ENABLE_G7_H32:-1}
ENABLE_G7_H32_WARM=${ENABLE_G7_H32_WARM:-1}
ENABLE_G7_H64=${ENABLE_G7_H64:-1}
ENABLE_B1_ENTROPY_LO=${ENABLE_B1_ENTROPY_LO:-1}
ENABLE_B1_ENTROPY_MID=${ENABLE_B1_ENTROPY_MID:-1}
ENABLE_B2_GINI_LO=${ENABLE_B2_GINI_LO:-1}

# ── 공통 고정 인자 ─────────────────────────────────────────
# G5_lowvar에서 검증된 최적 조건을 상속.
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-ff-loss 0.10
  --ff-alpha 1.0
  --weight-image 1.0
  --delta 0.05
"

# 방향 A (G7) 공통: spatial-gating 기반 (diversity loss 불필요)
ARGS_G7_BASE="
  ${BASE_ARGS}
  --spatial-gating
"

# 방향 B 공통: 전역 gating + diversity loss (G5_lowvar 조건 상속)
# --diversity-loss-space softmax: entropy/gini는 softmax 공간에서 정의
ARGS_B_BASE="
  ${BASE_ARGS}
  --weight-diversity-loss 0.01
  --diversity-loss-space softmax
  --gating-grad-clip 1.0
  --target-diversity-var 0.5
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ──────────────────────────────────
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
        echo "  Model → ${OUTPUT_MODEL}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# [방향 A] 세트 1: G7_h32
#
#   검증 목적: z_repr을 GatingMLP에 추가했을 때 G6_h32 대비 성능 변화.
#              content-aware gating의 핵심 가설 검증.
#   비교 대상: G6_h32 (기존 결과 재사용)
#   변수    : hidden_dim=32, z-gating=ON, warmup=0
#   주의    : Pass1 forward가 추가되어 학습 시간 약 1.5배 증가 예상.
# ================================================================
if [ "${ENABLE_G7_H32}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  [방향 A] 세트 1: G7_h32"
    echo "  --spatial-gating (z-gating 포함)"
    echo "  --gating-hidden-dim 32"
    echo "  --gating-warmup-iters 0"
    echo "  z_repr: density_net Pass1 피처, detach() 적용"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G7_h32" "${PATIENT}" "${MODALITY}" \
                ${ARGS_G7_BASE} \
                --gating-hidden-dim 32 \
                --gating-warmup-iters 0
        done
    done
else
    echo "  [SKIP] G7_h32 (ENABLE_G7_H32=0)"
fi

# ================================================================
# [방향 A] 세트 2: G7_h32_warm
#
#   검증 목적: density_net 초반 동결(500 iter) + z_repr gating 조합.
#              warmup 중에는 z_repr이 불안정하므로 gating도 초반엔
#              좌표만 보는 셈 — 점진적 안정화 효과 기대.
#   비교 대상: G7_h32 (warmup 유무만 다름)
#   변수    : hidden_dim=32, z-gating=ON, warmup=500
# ================================================================
if [ "${ENABLE_G7_H32_WARM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  [방향 A] 세트 2: G7_h32_warm"
    echo "  --spatial-gating (z-gating 포함)"
    echo "  --gating-hidden-dim 32"
    echo "  --gating-warmup-iters 500"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G7_h32_warm" "${PATIENT}" "${MODALITY}" \
                ${ARGS_G7_BASE} \
                --gating-hidden-dim 32 \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] G7_h32_warm (ENABLE_G7_H32_WARM=0)"
fi

# ================================================================
# [방향 A] 세트 3: G7_h64
#
#   검증 목적: z_repr(n_features_z=15) + 좌표(3) = 18차원 입력에서
#              hidden_dim=64가 더 풍부한 content-aware gating을 학습하는가.
#              GatingMLP params: Linear(18->64) + Linear(64->12) ≈ 1900개.
#   비교 대상: G7_h32 (hidden_dim만 다름), G6_h64 (z-gating 유무)
#   변수    : hidden_dim=64, z-gating=ON, warmup=0
#   주의    : OOM 위험 — h32 세트 이후 실행 (의도적 배치).
# ================================================================
if [ "${ENABLE_G7_H64}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  [방향 A] 세트 3: G7_h64"
    echo "  --spatial-gating (z-gating 포함)"
    echo "  --gating-hidden-dim 64"
    echo "  --gating-warmup-iters 0"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G7_h64" "${PATIENT}" "${MODALITY}" \
                ${ARGS_G7_BASE} \
                --gating-hidden-dim 64 \
                --gating-warmup-iters 0
        done
    done
else
    echo "  [SKIP] G7_h64 (ENABLE_G7_H64=0)"
fi

# ================================================================
# [방향 B] 세트 4: B1_entropy_lo
#
#   검증 목적: entropy 기반 diversity loss(부호 수정 버전)가
#              variance 기반 G5_lowvar와 성능 차이를 보이는가.
#              entropy는 softmax 공간에서 균등 분포일수록 높으므로
#              낮추는 방향이 곧 레벨 분화 유도.
#   비교 대상: G5_lowvar (diversity_fn=variance, 기존 결과 재사용)
#   변수    : fn=entropy, w=0.01 (낮은 weight — 안정성 우선)
#   주의    : --diversity-loss-space softmax + target=0.5
#             (softmax 공간 entropy 범위에 맞춘 target)
# ================================================================
if [ "${ENABLE_B1_ENTROPY_LO}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  [방향 B] 세트 4: B1_entropy_lo"
    echo "  --diversity-loss-fn entropy"
    echo "  --weight-diversity-loss 0.01"
    echo "  --diversity-loss-space softmax"
    echo "  --target-diversity-var 0.5"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "B1_entropy_lo" "${PATIENT}" "${MODALITY}" \
                ${ARGS_B_BASE} \
                --diversity-loss-fn entropy \
                --weight-diversity-loss 0.01
        done
    done
else
    echo "  [SKIP] B1_entropy_lo (ENABLE_B1_ENTROPY_LO=0)"
fi

# ================================================================
# [방향 B] 세트 5: B1_entropy_mid
#
#   검증 목적: entropy weight를 0.01→0.05로 높였을 때
#              분화가 더 강해지는지, 아니면 불안정해지는지.
#              G5_lowvar의 variance w=0.05와 직접 대응.
#   비교 대상: B1_entropy_lo (weight만 다름), G5_lowvar
#   변수    : fn=entropy, w=0.05 (G5_lowvar weight와 동일)
# ================================================================
if [ "${ENABLE_B1_ENTROPY_MID}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  [방향 B] 세트 5: B1_entropy_mid"
    echo "  --diversity-loss-fn entropy"
    echo "  --weight-diversity-loss 0.05"
    echo "  --diversity-loss-space softmax"
    echo "  --target-diversity-var 0.5"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "B1_entropy_mid" "${PATIENT}" "${MODALITY}" \
                ${ARGS_B_BASE} \
                --diversity-loss-fn entropy \
                --weight-diversity-loss 0.05
        done
    done
else
    echo "  [SKIP] B1_entropy_mid (ENABLE_B1_ENTROPY_MID=0)"
fi

# ================================================================
# [방향 B] 세트 6: B2_gini_lo
#
#   검증 목적: gini 기반 loss가 entropy 대비 어떤 특성을 보이는가.
#              gini는 entropy보다 계산이 단순하고 gradient가 부드러움.
#              동일 weight(0.01)로 entropy_lo와 직접 비교.
#   비교 대상: B1_entropy_lo (fn만 다름)
#   변수    : fn=gini, w=0.01
# ================================================================
if [ "${ENABLE_B2_GINI_LO}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  [방향 B] 세트 6: B2_gini_lo"
    echo "  --diversity-loss-fn gini"
    echo "  --weight-diversity-loss 0.01"
    echo "  --diversity-loss-space softmax"
    echo "  --target-diversity-var 0.5"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "B2_gini_lo" "${PATIENT}" "${MODALITY}" \
                ${ARGS_B_BASE} \
                --diversity-loss-fn gini \
                --weight-diversity-loss 0.01
        done
    done
else
    echo "  [SKIP] B2_gini_lo (ENABLE_B2_GINI_LO=0)"
fi

# ================================================================
# 평가 실행
# G5_lowvar, G6_h32 결과는 기존 파일을 재사용 (재실행 없이 비교)
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags G5_lowvar G6_h32 \
               G7_h32 G7_h32_warm G7_h64 \
               B1_entropy_lo B1_entropy_mid B2_gini_lo \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_G7B_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_G7B_sweep.csv"
fi

# ── 최종 요약 ────────────────────────────────────────────────
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

#!/usr/bin/env bash
# ================================================================
# run_G6_sweep.sh
# Gating G6 실험 스윗 — 공간 적응형 GatingMLP
#
# 대실험 배경:
#   G5_lowvar(w=0.05, tv=0.12, clip=0.0)이 현재 베스트.
#   전역 단일 level_weights 대신 위치별 GatingMLP를 사용하면
#   피질/백질 등 공간적으로 다른 구조가 각자에게 최적인
#   해시 레벨을 선택적으로 활용할 수 있다는 가설 검증.
#
#   핵심 질문:
#   Q1. GatingMLP(hidden_dim=32) 자체가 G5_lowvar를 이기는가?
#       → G6_h32 vs G5_lowvar(기존 결과 재사용)
#
#   Q2. warmup(density_net 초반 동결)이 GatingMLP 학습에 도움되는가?
#       → G6_h32_warm vs G6_h32
#
#   Q3. hidden_dim=64로 표현력을 높이면 성능이 더 올라가는가?
#       → G6_h64 vs G6_h32
#
#   Q4. hidden_dim=64 + warmup 조합이 실전 최적인가?
#       → G6_h64_warm vs G6_h64 / G6_h32_warm
#
# 실험 세트 (메모리 오름차순 — OOM 안전 순서):
#   G6_h32       hidden_dim=32  warmup=0    (~500  params)
#   G6_h32_warm  hidden_dim=32  warmup=500  (~500  params, 초반 density_net 동결)
#   G6_h64       hidden_dim=64  warmup=0    (~1100 params)
#   G6_h64_warm  hidden_dim=64  warmup=500  (~1100 params, 초반 density_net 동결)
#
# 비교 기준선:
#   G5_lowvar 결과는 기존 실행 결과를 재사용 (재실행 불필요)
#   evaluate_sweep.py 에 G5_lowvar 태그를 포함시켜 비교
#
# 메모리 비고:
#   GatingMLP 파라미터 자체는 작으나,
#   _apply_gating 내 repeat_interleave 후 weights_expanded
#   (N_px*n_samples, n_levels) ≈ 18432*256*12*4 bytes ≈ 216 MB 가
#   실질적 메모리 소비원. hidden_dim 차이보다 이 텐서가 지배적.
#   OOM 발생 시 h64 세트부터 위험 — 따라서 h32를 먼저 실행.
#
# 대상 환자: 003 026 030 040 060
# 모달리티 : flair / t2 / t1ce
# 총 케이스: 4세트 x 5명 x 3모달리티 = 60
#
# 사용법:
#   chmod +x run_G6_sweep.sh
#   ./run_G6_sweep.sh                         # 모든 세트 실행
#   ENABLE_H32=0      ./run_G6_sweep.sh       # G6_h32 건너뜀
#   ENABLE_H32_WARM=0 ./run_G6_sweep.sh       # G6_h32_warm 건너뜀
#   ENABLE_H64=0      ./run_G6_sweep.sh       # G6_h64 건너뜀
#   ENABLE_H64_WARM=0 ./run_G6_sweep.sh       # G6_h64_warm 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_H32=${ENABLE_H32:-1}
ENABLE_H32_WARM=${ENABLE_H32_WARM:-1}
ENABLE_H64=${ENABLE_H64:-1}
ENABLE_H64_WARM=${ENABLE_H64_WARM:-1}

# ── 공통 고정 인자 ─────────────────────────────────────────
# G5_lowvar에서 검증된 최적 조건을 상속.
# spatial_gating=True 시 전역 diversity loss는 train.py에서
# 자동 비활성화되므로 --weight-diversity-loss 인자 불필요.
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
  --spatial-gating
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
        ${BASE_ARGS} \
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
# 세트 1: G6_h32  [메모리 최소 — 가장 먼저 실행]
#
#   검증 목적: GatingMLP(hidden_dim=32) 자체 효과
#              G5_lowvar(전역 gating 베스트) 대비 공간 적응형이
#              실제로 성능을 높이는지 확인하는 핵심 실험.
#   비교 대상: G5_lowvar (기존 결과 재사용)
#   변수    : hidden_dim=32, warmup=0
# ================================================================
if [ "${ENABLE_H32}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: G6_h32"
    echo "  --spatial-gating"
    echo "  --gating-hidden-dim 32"
    echo "  --gating-warmup-iters 0  (warmup 없음)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G6_h32" "${PATIENT}" "${MODALITY}" \
                --gating-hidden-dim 32 \
                --gating-warmup-iters 0
        done
    done
else
    echo "  [SKIP] G6_h32 (ENABLE_H32=0)"
fi

# ================================================================
# 세트 2: G6_h32_warm  [메모리 최소 — warmup 효과 분리]
#
#   검증 목적: density_net 초반 동결(500 iter)이 GatingMLP의
#              공간 패턴 학습을 돕는지 확인.
#              warmup 중 density_net grad graph가 없어
#              초반 메모리는 G6_h32보다 오히려 낮음.
#   비교 대상: G6_h32 (warmup 유무만 다름)
#   변수    : hidden_dim=32, warmup=500
# ================================================================
if [ "${ENABLE_H32_WARM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: G6_h32_warm"
    echo "  --spatial-gating"
    echo "  --gating-hidden-dim 32"
    echo "  --gating-warmup-iters 500"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G6_h32_warm" "${PATIENT}" "${MODALITY}" \
                --gating-hidden-dim 32 \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] G6_h32_warm (ENABLE_H32_WARM=0)"
fi

# ================================================================
# 세트 3: G6_h64  [메모리 중간 — 표현력 상한 확인]
#
#   검증 목적: hidden_dim을 32→64로 높였을 때 성능이 올라가는지.
#              공간 패턴 학습에 더 많은 표현력이 필요한지 판단.
#              GatingMLP params: ~1100개 (h32의 약 2배).
#   비교 대상: G6_h32 (hidden_dim만 다름)
#   변수    : hidden_dim=64, warmup=0
# ================================================================
if [ "${ENABLE_H64}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: G6_h64"
    echo "  --spatial-gating"
    echo "  --gating-hidden-dim 64"
    echo "  --gating-warmup-iters 0  (warmup 없음)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G6_h64" "${PATIENT}" "${MODALITY}" \
                --gating-hidden-dim 64 \
                --gating-warmup-iters 0
        done
    done
else
    echo "  [SKIP] G6_h64 (ENABLE_H64=0)"
fi

# ================================================================
# 세트 4: G6_h64_warm  [메모리 최대 — 마지막 실행]
#
#   검증 목적: hidden_dim=64 + warmup=500 조합이
#              지금까지의 모든 세트를 이기는지 확인.
#              Q1~Q3에서 긍정 결과가 나왔을 때의 최종 후보.
#              OOM 위험이 가장 높으므로 의도적으로 마지막 배치.
#   비교 대상: G6_h64 (warmup 유무), G6_h32_warm (hidden_dim 차이)
#   변수    : hidden_dim=64, warmup=500
# ================================================================
if [ "${ENABLE_H64_WARM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 4: G6_h64_warm"
    echo "  --spatial-gating"
    echo "  --gating-hidden-dim 64"
    echo "  --gating-warmup-iters 500"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G6_h64_warm" "${PATIENT}" "${MODALITY}" \
                --gating-hidden-dim 64 \
                --gating-warmup-iters 500
        done
    done
else
    echo "  [SKIP] G6_h64_warm (ENABLE_H64_WARM=0)"
fi

# ================================================================
# 평가 실행
# G5_lowvar 결과는 기존 파일을 재사용 (재실행 없이 비교 가능)
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags H3 G5_lowvar G6_h32 G6_h32_warm G6_h64 G6_h64_warm \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_G6_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_G6_sweep.csv"
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

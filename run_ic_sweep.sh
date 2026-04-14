#!/usr/bin/env bash
# ================================================================
# run_ic_sweep.sh  (v1 — 2026-04-14)
# IC 베이스 기반 n_iter × gamma × milestones 탐색
#
# ── 실험 배경 ──────────────────────────────────────────────────
# time_sweep 결과 요약:
#   ff_w005_a20 (BASE, n_iter=6000) : PSNR 25.726
#   time_IC     (n_iter=2000)       : PSNR 26.143  ← 현재 최고, 새 베이스
#   time_Z      (n_iter=3000, gamma=0.10) : PSNR 25.973
#                                    → gamma=0.1이 동일 시간 대비 +0.083 확인
#
# ── 핵심 가설 ──────────────────────────────────────────────────
# H1. n_iter=2000(IC)에서도 과적합/포화가 진행 중
#     → 1500, 1000으로 더 줄이면 추가 개선 가능
# H2. gamma=0.1 (강한 lr decay)이 짧은 학습에 유리
#     → time_Z에서 이미 검증, IC에서도 가산적 개선 기대
# H3. milestone 비율 조정이 n_iter=2000에서 효과적일 수 있음
#     → 단, 효과가 미미하면 노이즈로 처리하고 채택 안 함
#
# ── 탐색 설계 ──────────────────────────────────────────────────
# 2×2 factorial (n_iter × gamma) + milestone 변형 1개:
#
#               gamma=0.33          gamma=0.10
#   n_iter=2000   IC (기존)          IC-Z  (신규)
#   n_iter=1500   IC-2 (신규)        IC-2Z (신규)
#   n_iter=1000   IC-3 (신규)        IC-3Z (신규)
#
#   IC-M: n_iter=2000, gamma=0.33, milestones=[0.4, 0.70, 0.9]
#         IC와 gamma/n_iter 동일, milestone 구조만 변경 → 순수 비교
#         채택 기준: 노이즈 수준이면 폐기
#
# ── lr 궤적 비교 ───────────────────────────────────────────────
# gamma=0.33, n_iter=2000, milestones=[0.5, 0.75, 0.9]:
#   iter 1000: 5e-3 → 1.65e-3
#   iter 1500: 1.65e-3 → 5.4e-4
#   iter 1800: 5.4e-4 → 1.8e-4
#
# gamma=0.10, n_iter=2000, milestones=[0.5, 0.75, 0.9]:
#   iter 1000: 5e-3 → 5e-4
#   iter 1500: 5e-4 → 5e-5
#   iter 1800: 5e-5 → 5e-6   ← 마지막 200iter 극저 lr fine-tuning
#
# gamma=0.33, n_iter=1500, milestones=[0.5, 0.75, 0.9]:
#   iter  750: 5e-3 → 1.65e-3
#   iter 1125: 1.65e-3 → 5.4e-4
#   iter 1350: 5.4e-4 → 1.8e-4
#
# gamma=0.10, n_iter=1000, milestones=[0.5, 0.75, 0.9]:
#   iter  500: 5e-3 → 5e-4
#   iter  750: 5e-4 → 5e-5
#   iter  900: 5e-5 → 5e-6
#
# gamma=0.33, n_iter=2000, milestones=[0.4, 0.70, 0.9] (IC-M):
#   iter  800: 5e-3 → 1.65e-3   (IC 대비 200iter 빠름)
#   iter 1400: 1.65e-3 → 5.4e-4 (IC 대비 100iter 빠름)
#   iter 1800: 5.4e-4 → 1.8e-4  (동일)
#
# ── 평가 구조 ──────────────────────────────────────────────────
#   time_IC   : 기존 결과 재사용 (재실행 없음)
#   신규 6세트 + IC 합계 7세트 비교
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q1. n_iter=2000보다 1500이 더 좋은가?           → IC vs IC-2
# Q2. n_iter=1000까지 단조증가가 이어지는가?       → IC-2 vs IC-3
# Q3. gamma=0.1이 IC에서도 유효한가?              → IC vs IC-Z
# Q4. n_iter 단축 + gamma 강화는 가산적인가?       → IC-2Z vs IC-2 vs IC-Z
# Q5. IC-3에서도 gamma=0.1이 유효한가?            → IC-3 vs IC-3Z
# Q6. milestone 조정의 효과는 노이즈 수준인가?     → IC vs IC-M
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 신규 6세트 × 15 = 90  (평가: 7세트 × 15 = 105)
#
# 사용법:
#   chmod +x run_ic_sweep.sh
#   ./run_ic_sweep.sh                      # 모든 신규 세트 실행 후 평가
#   ENABLE_IC2=0  ./run_ic_sweep.sh        # IC-2  건너뜀
#   ENABLE_ICZ=0  ./run_ic_sweep.sh        # IC-Z  건너뜀
#   ENABLE_IC2Z=0 ./run_ic_sweep.sh        # IC-2Z 건너뜀
#   ENABLE_IC3=0  ./run_ic_sweep.sh        # IC-3  건너뜀
#   ENABLE_IC3Z=0 ./run_ic_sweep.sh        # IC-3Z 건너뜀
#   ENABLE_ICM=0  ./run_ic_sweep.sh        # IC-M  건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_IC2=${ENABLE_IC2:-1}
ENABLE_ICZ=${ENABLE_ICZ:-1}
ENABLE_IC2Z=${ENABLE_IC2Z:-1}
ENABLE_IC3=${ENABLE_IC3:-1}
ENABLE_IC3Z=${ENABLE_IC3Z:-1}
ENABLE_ICM=${ENABLE_ICM:-1}

# ── BASE_ARGS: time_IC 완전 재현 인자 ─────────────────────────
# --n-iter, --gamma, --milestones 는 세트별로 명시적으로 덮어씀
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
  --batch-size 4096
  --learning-rate 5e-3
  --seed 42
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
# 세트 IC-2: n_iter=1500, gamma=0.33
#
#   목적: IC(2000)에서 단조증가 패턴이 이어지는지 확인.
#         1500에서 꺾이면 → 최적점이 1500~2000 사이.
#         1500에서도 오르면 → IC-3(1000) 탐색 근거.
#   변경: --n-iter 1500  (gamma, milestones는 IC와 동일)
# ================================================================
if [ "${ENABLE_IC2}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 IC-2: n_iter=1500, gamma=0.33, milestones=[0.5,0.75,0.9]"
    echo "  비교: IC (n_iter=2000) vs IC-2 → n_iter 단축 순효과"
    echo "  lr 궤적: 5e-3 → 1.65e-3(750) → 5.4e-4(1125) → 1.8e-4(1350)"
    echo "  예상 시간: ~200초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IC2" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 1500 \
                --gamma 0.33 \
                --milestones 0.5 0.75 0.9
        done
    done
else
    echo "  [SKIP] IC-2 (ENABLE_IC2=0)"
fi

# ================================================================
# 세트 IC-Z: n_iter=2000, gamma=0.10
#
#   목적: time_Z(n_iter=3000, gamma=0.1)에서 확인된 gamma 효과가
#         IC(n_iter=2000)에서도 성립하는지 검증.
#         IC와 n_iter 완전 동일, gamma만 변경 → 순수 비교.
#   변경: --gamma 0.10
# ================================================================
if [ "${ENABLE_ICZ}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 IC-Z: n_iter=2000, gamma=0.10, milestones=[0.5,0.75,0.9]"
    echo "  비교: IC (gamma=0.33) vs IC-Z (gamma=0.10) → gamma 단독 효과"
    echo "  lr 궤적: 5e-3 → 5e-4(1000) → 5e-5(1500) → 5e-6(1800)"
    echo "  예상 시간: ~270초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_ICZ" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 2000 \
                --gamma 0.10 \
                --milestones 0.5 0.75 0.9
        done
    done
else
    echo "  [SKIP] IC-Z (ENABLE_ICZ=0)"
fi

# ================================================================
# 세트 IC-2Z: n_iter=1500, gamma=0.10
#
#   목적: n_iter 단축(IC-2)과 gamma 강화(IC-Z)가 가산적인지 확인.
#         IC-2Z > IC-2 + IC-Z 개선합 → 상호작용 시너지.
#         IC-2Z ≈ IC-2 or IC-Z 중 하나 → 한 효과가 지배적.
#   변경: --n-iter 1500 / --gamma 0.10
# ================================================================
if [ "${ENABLE_IC2Z}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 IC-2Z: n_iter=1500, gamma=0.10, milestones=[0.5,0.75,0.9]"
    echo "  비교: IC-2(gamma=0.33) vs IC-2Z → 1500에서 gamma 효과"
    echo "  비교: IC-Z(n_iter=2000) vs IC-2Z → gamma=0.1에서 n_iter 효과"
    echo "  lr 궤적: 5e-3 → 5e-4(750) → 5e-5(1125) → 5e-6(1350)"
    echo "  예상 시간: ~200초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IC2Z" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 1500 \
                --gamma 0.10 \
                --milestones 0.5 0.75 0.9
        done
    done
else
    echo "  [SKIP] IC-2Z (ENABLE_IC2Z=0)"
fi

# ================================================================
# 세트 IC-3: n_iter=1000, gamma=0.33
#
#   목적: n_iter 하한 탐색. IC-2(1500)에서도 개선이 이어질 경우
#         1000까지의 패턴을 확인.
#         주의: milestone 절대값이 iter 500/750/900으로 매우 이름.
#         학습 극초반부터 lr이 빠르게 낮아지는 구조 — 불안정 가능성 있음.
#   변경: --n-iter 1000
# ================================================================
if [ "${ENABLE_IC3}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 IC-3: n_iter=1000, gamma=0.33, milestones=[0.5,0.75,0.9]"
    echo "  비교: IC-2 (n_iter=1500) vs IC-3 → 하한 탐색"
    echo "  lr 궤적: 5e-3 → 1.65e-3(500) → 5.4e-4(750) → 1.8e-4(900)"
    echo "  주의: milestone 절대값이 이르므로 학습 불안정 가능성 모니터링"
    echo "  예상 시간: ~135초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IC3" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 1000 \
                --gamma 0.33 \
                --milestones 0.5 0.75 0.9
        done
    done
else
    echo "  [SKIP] IC-3 (ENABLE_IC3=0)"
fi

# ================================================================
# 세트 IC-3Z: n_iter=1000, gamma=0.10
#
#   목적: 최단 학습(1000iter)에서 강한 decay(gamma=0.1) 조합.
#         gamma=0.1이면 iter 500에서 lr이 5e-4로 급감하므로
#         학습 후반부 대부분이 극저 lr fine-tuning 구간이 됨.
#         IC-3(gamma=0.33)와 비교하면 1000iter에서 gamma 효과 분리.
#   변경: --n-iter 1000 / --gamma 0.10
# ================================================================
if [ "${ENABLE_IC3Z}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 IC-3Z: n_iter=1000, gamma=0.10, milestones=[0.5,0.75,0.9]"
    echo "  비교: IC-3 (gamma=0.33) vs IC-3Z (gamma=0.10)"
    echo "  lr 궤적: 5e-3 → 5e-4(500) → 5e-5(750) → 5e-6(900)"
    echo "  예상 시간: ~135초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IC3Z" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 1000 \
                --gamma 0.10 \
                --milestones 0.5 0.75 0.9
        done
    done
else
    echo "  [SKIP] IC-3Z (ENABLE_IC3Z=0)"
fi

# ================================================================
# 세트 IC-M: n_iter=2000, gamma=0.33, milestones=[0.4, 0.70, 0.9]
#
#   목적: milestone 비율 조정의 순효과 확인.
#         IC와 n_iter/gamma 완전 동일, milestone 구조만 변경.
#         lr을 더 일찍 낮춰 중반부 안정화를 앞당김.
#   변경: --milestones 0.4 0.70 0.9
#   lr 궤적:
#     IC   [0.5,0.75,0.9]: 5e-3 → 1.65e-3(1000) → 5.4e-4(1500) → 1.8e-4(1800)
#     IC-M [0.4,0.70,0.9]: 5e-3 → 1.65e-3( 800) → 5.4e-4(1400) → 1.8e-4(1800)
#   채택 기준: IC 대비 유의미한 개선이 아니면 노이즈로 처리하고 폐기.
# ================================================================
if [ "${ENABLE_ICM}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 IC-M: n_iter=2000, gamma=0.33, milestones=[0.4,0.70,0.9]"
    echo "  비교: IC (milestones=[0.5,0.75,0.9]) vs IC-M → milestone 순효과"
    echo "  lr 궤적: 5e-3 → 1.65e-3(800) → 5.4e-4(1400) → 1.8e-4(1800)"
    echo "  채택 기준: 눈에 띄는 개선 없으면 노이즈로 처리, 폐기"
    echo "  예상 시간: ~270초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_ICM" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 2000 \
                --gamma 0.33 \
                --milestones 0.4 0.70 0.9
        done
    done
else
    echo "  [SKIP] IC-M (ENABLE_ICM=0)"
fi

# ================================================================
# 평가 실행
# time_IC : 기존 결과 재사용 (재실행 없음)
# 총 7세트 비교
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  총 7세트: time_IC / IC2 / ICZ / IC2Z / IC3 / IC3Z / ICM"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags time_IC \
               time_IC2 \
               time_ICZ \
               time_IC2Z \
               time_IC3 \
               time_IC3Z \
               time_ICM \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_ic_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_ic_sweep.csv"
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

#!/usr/bin/env bash
# ================================================================
# run_hsm_dataset.sh  (v1 — 2026-04-30)
# Hard Slice Mining alpha 탐색 — 48명 전체 데이터셋 생성
#
# ── start_iter=0 고정, alpha만 변화 ───────────────────────────
#   근거: 2차 탐색에서 alpha 낮은 구간(0.7, 0.8)에서
#         start=0 ≥ start=500 일관 확인 → start_iter 변수 제거
#
# ── 2차 탐색 결과 요약 (5명 기준) ────────────────────────────
#   hsm_I (alpha=0.8, start=0): 전체 PSNR 26.343, T1CE +0.059 vs IC
#   hsm_L (alpha=0.7, start=0): 전체 SSIM/LPIPS 최우수
#
# ── 탐색 세트 (start=0 고정) ──────────────────────────────────
#   hsm_I : alpha=0.80  (2차 best   — 기존 5명 복사 가능)
#   hsm_M : alpha=0.75  (I-L 보간)
#   hsm_L : alpha=0.70  (2차 준best — 기존 5명 복사 가능)
#   hsm_N : alpha=0.60  (하락 경향 확인)
#   hsm_O : alpha=0.50  (극단 탐색)
#   hsm_P : alpha=0.40  (경계 확인)
#
# ── EMA 잔존율 (k=19.9 실측 기준) ────────────────────────────
#   alpha=0.80 →  1.2%
#   alpha=0.75 →  0.3%
#   alpha=0.70 → ~0.1%
#   alpha=0.60 → ~0.0%
#   alpha=0.50 → ~0.0%
#   alpha=0.40 → ~0.0%
#
# ── 출력 구조 ─────────────────────────────────────────────────
#   recon_HSM_hsm_I/
#     BraTS20_Training_003/
#       003_flair_4x5_hsm_I.nii.gz
#       003_t1ce_4x5_hsm_I.nii.gz
#     ...
#
# ── 예상 시간 ─────────────────────────────────────────────────
#   세트당: 43명 신규 × 2모달리티 × 260초 ≈ 6.2시간
#   (hsm_I, hsm_L: 기존 5명 복사 → 실 학습 43명)
#   (hsm_M, hsm_N, hsm_O, hsm_P: 전원 신규 학습 48명 × 2 × 260초 ≈ 6.9시간)
#   전체 6세트 합계: 약 40시간
#
# 사용법:
#   chmod +x run_hsm_dataset.sh
#   ./run_hsm_dataset.sh                       # 전체 순차 실행 + 평가
#   ENABLE_HSM_I=0 ./run_hsm_dataset.sh        # hsm_I 건너뜀
#   ENABLE_HSM_M=0 ./run_hsm_dataset.sh        # hsm_M 건너뜀
#   ENABLE_HSM_L=0 ./run_hsm_dataset.sh        # hsm_L 건너뜀
#   ENABLE_HSM_N=0 ./run_hsm_dataset.sh        # hsm_N 건너뜀
#   ENABLE_HSM_O=0 ./run_hsm_dataset.sh        # hsm_O 건너뜀
#   ENABLE_HSM_P=0 ./run_hsm_dataset.sh        # hsm_P 건너뜀
#   SKIP_EVAL=1    ./run_hsm_dataset.sh        # 평가 건너뜀
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_HSM_I=${ENABLE_HSM_I:-1}
ENABLE_HSM_M=${ENABLE_HSM_M:-1}
ENABLE_HSM_L=${ENABLE_HSM_L:-1}
ENABLE_HSM_N=${ENABLE_HSM_N:-1}
ENABLE_HSM_O=${ENABLE_HSM_O:-1}
ENABLE_HSM_P=${ENABLE_HSM_P:-1}
SKIP_EVAL=${SKIP_EVAL:-0}

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_BASE="/dshome/ddualab/dongnyeok/NeSVoR"
SWEEP_DIR="${RECON_BASE}"          # 기존 5명 sweep 결과 위치 (hsm_I, hsm_L)
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="${RECON_BASE}/test.csv"

# ── 공통 학습 설정 ─────────────────────────────────────────────
INTENSITY_MEAN=308
MODALITIES=(flair t1ce)
# hsm_I, hsm_L은 5명 sweep 결과가 SWEEP_DIR에 이미 존재
EXISTING_PATIENTS=(003 026 030 040 060)

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

# ── 전체 실패 누적 ────────────────────────────────────────────
ALL_FAILED=()

# ================================================================
# 세트 실행 함수
#   $1 = TAG (예: hsm_I)
#   $2 = ALPHA (예: 0.8)
#   $3 = HAS_EXISTING (1: 기존 5명 복사 가능, 0: 전원 신규 학습)
# ================================================================
run_dataset_set() {
    local TAG=$1
    local ALPHA=$2
    local HAS_EXISTING=$3

    local RECON_ROOT="${RECON_BASE}/recon_HSM_${TAG}"
    local FAILED_CASES=()
    local TOTAL=0
    local SUCCESS=0

    echo ""
    echo "################################################################"
    echo "  세트 시작: ${TAG}  (alpha=${ALPHA}, start_iter=0)"
    echo "  출력 루트: ${RECON_ROOT}"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    mkdir -p "${RECON_ROOT}"

    # ── 단계 1: 기존 5명 복사 ──────────────────────────────────
    if [ "${HAS_EXISTING}" -eq 1 ]; then
        echo ""
        echo "  [단계 1] 기존 5명 복사 (sweep 결과 재사용)"
        for PATIENT in "${EXISTING_PATIENTS[@]}"; do
            PATIENT_DIR="${RECON_ROOT}/BraTS20_Training_${PATIENT}"
            mkdir -p "${PATIENT_DIR}"
            for MODALITY in "${MODALITIES[@]}"; do
                SRC="${SWEEP_DIR}/${PATIENT}_${MODALITY}_4x5_${TAG}.nii.gz"
                DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${TAG}.nii.gz"
                if [ -f "${SRC}" ]; then
                    cp "${SRC}" "${DST}"
                    echo "    ✅ 복사: ${PATIENT} / ${MODALITY}"
                else
                    echo "    ⚠️  원본 없음 → 신규 학습 대상으로 전환: ${PATIENT} / ${MODALITY}"
                    # 단계 2에서 DST 파일이 없으면 자동으로 학습함
                fi
            done
        done
    fi

    # ── 단계 2: test.csv 기반 순차 학습 ──────────────────────────
    echo ""
    echo "  [단계 2] test.csv 기반 순차 학습 (alpha=${ALPHA}, start_iter=0)"

    mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

    for FULL_ID in "${FULL_IDS[@]}"; do
        PATIENT="${FULL_ID##*_}"
        PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
        mkdir -p "${PATIENT_DIR}"

        for MODALITY in "${MODALITIES[@]}"; do
            DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_${TAG}.nii.gz"

            # 이미 존재하면 스킵 (단계 1 복사분 포함)
            if [ -f "${DST}" ]; then
                echo "    [SKIP] 이미 존재: ${PATIENT} / ${MODALITY}"
                continue
            fi

            INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

            if [ ! -f "${INPUT}" ]; then
                echo "    ❌ 입력 없음: ${PATIENT} / ${MODALITY}  (${INPUT})"
                FAILED_CASES+=("${TAG}_${PATIENT}_${MODALITY}_input_missing")
                TOTAL=$((TOTAL + 1))
                continue
            fi

            echo ""
            echo "  ================================================================"
            echo "  [START] ${TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            echo "  ================================================================"

            # shellcheck disable=SC2086
            nesvor reconstruct \
                --input-stacks  "${INPUT}" \
                --output-volume "${DST}" \
                --sample-mask   "${INPUT}" \
                --output-intensity-mean ${INTENSITY_MEAN} \
                ${BASE_ARGS} \
                --slice-residual-alpha "${ALPHA}" \
                --hard-mining-start-iter 0

            EXIT_CODE=$?
            TOTAL=$((TOTAL + 1))
            if [ ${EXIT_CODE} -ne 0 ]; then
                echo "  ❌ FAILED: ${TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
                FAILED_CASES+=("${TAG}_${PATIENT}_${MODALITY}")
            else
                SUCCESS=$((SUCCESS + 1))
                echo "  ✅ DONE : ${TAG} / ${PATIENT} / ${MODALITY}"
                echo "       → ${DST}"
                echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            fi
        done
    done

    # 세트 요약
    echo ""
    echo "  ── 세트 ${TAG} 완료: $(date '+%Y-%m-%d %H:%M:%S') ──"
    echo "  학습 성공: ${SUCCESS} / ${TOTAL}"
    if [ ${#FAILED_CASES[@]} -gt 0 ]; then
        echo "  ❌ 실패 (${#FAILED_CASES[@]}건):"
        for c in "${FAILED_CASES[@]}"; do
            echo "       - ${c}"
        done
        ALL_FAILED+=("${FAILED_CASES[@]}")
    else
        echo "  ✅ 모든 케이스 성공"
    fi
}

# ================================================================
# 세트별 순차 실행
# ================================================================

if [ "${ENABLE_HSM_I}" -eq 1 ]; then
    run_dataset_set "hsm_I" "0.8" "1"   # 기존 5명 복사 가능
else
    echo "  [SKIP] hsm_I (ENABLE_HSM_I=0)"
fi

if [ "${ENABLE_HSM_M}" -eq 1 ]; then
    run_dataset_set "hsm_M" "0.75" "0"  # 전원 신규 학습
else
    echo "  [SKIP] hsm_M (ENABLE_HSM_M=0)"
fi

if [ "${ENABLE_HSM_L}" -eq 1 ]; then
    run_dataset_set "hsm_L" "0.7" "1"   # 기존 5명 복사 가능
else
    echo "  [SKIP] hsm_L (ENABLE_HSM_L=0)"
fi

if [ "${ENABLE_HSM_N}" -eq 1 ]; then
    run_dataset_set "hsm_N" "0.6" "0"
else
    echo "  [SKIP] hsm_N (ENABLE_HSM_N=0)"
fi

if [ "${ENABLE_HSM_O}" -eq 1 ]; then
    run_dataset_set "hsm_O" "0.5" "0"
else
    echo "  [SKIP] hsm_O (ENABLE_HSM_O=0)"
fi

if [ "${ENABLE_HSM_P}" -eq 1 ]; then
    run_dataset_set "hsm_P" "0.4" "0"
else
    echo "  [SKIP] hsm_P (ENABLE_HSM_P=0)"
fi

# ================================================================
# 평가 실행
# evaluate_batch_v2.py는 설정부를 직접 수정하는 구조이므로
# 래퍼 스크립트 eval_hsm_dataset.py를 별도 실행
# ================================================================
if [ "${SKIP_EVAL}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  평가 시작: eval_hsm_dataset.py"
    echo "  대상: IC(기준) + HSM 6세트 (flair, t1ce)"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    python eval_hsm_dataset.py

    EVAL_EXIT=$?
    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
    else
        echo "  ✅ 평가 완료 → eval_results/eval_hsm_dataset.csv"
    fi
else
    echo "  [SKIP] 평가 (SKIP_EVAL=1)"
fi

# ── 전체 최종 요약 ────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${#ALL_FAILED[@]} -eq 0 ]; then
    echo "  ✅ 모든 세트 성공"
else
    echo "  ❌ 전체 실패 케이스 (${#ALL_FAILED[@]}건):"
    for c in "${ALL_FAILED[@]}"; do
        echo "       - ${c}"
    done
fi
echo "================================================================"

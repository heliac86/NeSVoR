#!/usr/bin/env bash
# ================================================================
# run_ic_dataset.sh  (v1 — 2026-04-14)
# IC 설정으로 test셋 48명 전체 재구성 → 다운스트림 데이터셋 생성
#
# ── IC 확정 설정 ───────────────────────────────────────────────
#   n_iter=2000, batch=4096, lr=5e-3, gamma=0.33
#   milestones=[0.5, 0.75, 0.9]
#   weight-ff-loss=0.05, ff-alpha=2.0
#   PSNR 26.143 / SSIM 0.95916 / LPIPS 0.02388  (15 cases 평균)
#
# ── 처리 방식 ─────────────────────────────────────────────────
#   [단계 1] 기존 5명 (003 026 030 040 060):
#            time_IC 결과 파일을 IC 명명 규칙으로 rename 후
#            새 폴더 구조(recon_IC/)로 복사 (원본 보존)
#   [단계 2] 신규 43명:
#            test.csv에서 환자 ID 읽어 순차 학습
#            결과를 recon_IC/ 폴더 구조에 바로 저장
#
# ── 출력 구조 ─────────────────────────────────────────────────
#   recon_IC/
#     BraTS20_Training_003/
#       003_flair_4x5_IC.nii.gz
#       003_t1ce_4x5_IC.nii.gz
#     BraTS20_Training_026/
#       ...
#
# ── 입력 경로 ─────────────────────────────────────────────────
#   test.csv      : /dshome/ddualab/dongnyeok/NeSVoR/test.csv
#   degraded 원본 : /data/BraTS20_Degraded_4x_5/
#   기존 IC 파일  : /dshome/ddualab/dongnyeok/NeSVoR/
#
# 사용법:
#   chmod +x run_ic_dataset.sh
#   ./run_ic_dataset.sh                    # 전체 실행 (단계1+2)
#   SKIP_COPY=1 ./run_ic_dataset.sh        # 단계1(복사) 건너뜀
#   SKIP_TRAIN=1 ./run_ic_dataset.sh       # 단계2(학습) 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_IC.nii.gz
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 경로 설정 ─────────────────────────────────────────────────
RECON_ROOT="/dshome/ddualab/dongnyeok/NeSVoR/recon_IC"
EXISTING_DIR="/dshome/ddualab/dongnyeok/NeSVoR"
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"
TEST_CSV="/dshome/ddualab/dongnyeok/NeSVoR/test.csv"

# ── 플래그 ───────────────────────────────────────────────────
SKIP_COPY=${SKIP_COPY:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}

# ── 기존 결과 재활용 대상 5명 ─────────────────────────────────
EXISTING_PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t1ce)

# ── IC 학습 설정 ──────────────────────────────────────────────
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
  --weight-ff-loss 0.05
  --ff-alpha 2.0
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
# 단계 1: 기존 5명 파일 복사 (time_IC → IC, 새 폴더 구조로)
# ================================================================
if [ "${SKIP_COPY}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  단계 1: 기존 5명 파일 복사"
    echo "  time_IC → IC (rename) + recon_IC/ 폴더 구조로 복사"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    for PATIENT in "${EXISTING_PATIENTS[@]}"; do
        PATIENT_DIR="${RECON_ROOT}/BraTS20_Training_${PATIENT}"
        mkdir -p "${PATIENT_DIR}"

        for MODALITY in "${MODALITIES[@]}"; do
            SRC="${EXISTING_DIR}/${PATIENT}_${MODALITY}_4x5_time_IC.nii.gz"
            DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_IC.nii.gz"

            if [ -f "${SRC}" ]; then
                cp "${SRC}" "${DST}"
                echo "  ✅ 복사 완료: ${PATIENT} / ${MODALITY}"
                echo "     ${SRC}"
                echo "     → ${DST}"
            else
                echo "  ⚠️  원본 없음 (학습 대상으로 전환): ${PATIENT} / ${MODALITY}"
                echo "     찾는 경로: ${SRC}"
                # 원본이 없으면 학습 대상 목록에 추가하기 위해 플래그 파일 생성
                touch "${PATIENT_DIR}/.needs_train_${MODALITY}"
            fi
        done
    done

    echo ""
    echo "  단계 1 완료: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "  [SKIP] 단계 1 (SKIP_COPY=1)"
fi

# ================================================================
# 단계 2: 신규 환자 학습 + 기존 환자 중 원본 없는 케이스 보완
# ================================================================
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    echo ""
    echo "################################################################"
    echo "  단계 2: test.csv 기반 순차 학습"
    echo "  CSV : ${TEST_CSV}"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"

    # csv에서 Brats20ID 칼럼 읽기 (헤더 제외)
    # 형식: BraTS20_Training_151 → PATIENT_ID=151
    mapfile -t FULL_IDS < <(tail -n +2 "${TEST_CSV}" | cut -d',' -f1 | tr -d '\r')

    for FULL_ID in "${FULL_IDS[@]}"; do
        # BraTS20_Training_003 → PATIENT=003
        PATIENT="${FULL_ID##*_}"
        PATIENT_DIR="${RECON_ROOT}/${FULL_ID}"
        mkdir -p "${PATIENT_DIR}"

        for MODALITY in "${MODALITIES[@]}"; do
            DST="${PATIENT_DIR}/${PATIENT}_${MODALITY}_4x5_IC.nii.gz"

            # 이미 파일이 존재하면 스킵 (단계 1에서 복사된 경우 포함)
            if [ -f "${DST}" ]; then
                echo ""
                echo "  [SKIP] 이미 존재: ${PATIENT} / ${MODALITY}"
                echo "         ${DST}"
                continue
            fi

            # 단계 1에서 원본 없음 플래그가 있으면 제거 후 학습
            FLAG="${PATIENT_DIR}/.needs_train_${MODALITY}"
            [ -f "${FLAG}" ] && rm -f "${FLAG}"

            INPUT="${DEGRADED_ROOT}/${FULL_ID}/${FULL_ID}_${MODALITY}.nii"

            # 입력 파일 존재 확인
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
            echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            echo "================================================================"

            # shellcheck disable=SC2086
            nesvor reconstruct \
                --input-stacks  "${INPUT}" \
                --output-volume "${DST}" \
                --sample-mask   "${INPUT}" \
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
                echo "  Vol   → ${DST}"
                echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
            fi
        done
    done

    echo ""
    echo "  단계 2 완료: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "  [SKIP] 단계 2 (SKIP_TRAIN=1)"
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

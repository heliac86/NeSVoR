from argparse import Namespace
from typing import List, Tuple
import time
import datetime
import torch
import torch.optim as optim
import logging
from ..utils import MovingAverage, log_params, TrainLogger
# ===== [FF Loss 추가] 모델 임포트에 FF_LOSS 키 추가 =====
from .models import INR, NeSVoR, D_LOSS, S_LOSS, DS_LOSS, I_REG, B_REG, T_REG, D_REG, FF_LOSS, SLICE_MSE_KEY
# ===== [FF Loss 추가 끝] =====
from ..transform import RigidTransform
from ..image import Volume, Slice
from .data import PointDataset


def train(slices: List[Slice], args: Namespace) -> Tuple[INR, List[Slice], Volume]:
    # create training dataset
    dataset = PointDataset(slices)
    if args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.numel() // args.batch_size)

    use_scaling = True
    use_centering = True
    # perform centering and scaling
    spatial_scaling = 30.0 if use_scaling else 1
    bb = dataset.bounding_box
    center = (bb[0] + bb[-1]) / 2 if use_centering else torch.zeros_like(bb[0])
    ax = (
        RigidTransform(torch.cat([torch.zeros_like(center), -center])[None])
        .compose(dataset.transformation)
        .axisangle()
    )
    ax[:, -3:] /= spatial_scaling
    transformation = RigidTransform(ax)
    dataset.xyz /= spatial_scaling

    model = NeSVoR(
        transformation,
        dataset.resolution / spatial_scaling,
        dataset.mean,
        (bb - center) / spatial_scaling,
        spatial_scaling,
        args,
    )

    # ===== [k_norm] 훈련 타겟 정규화 =====
    v_mean = dataset.mean
    logging.info("[k_norm] Training target normalization: v_mean = %.4f", v_mean)
    dataset.normalize(v_mean)
    model.inr.v_mean.fill_(v_mean)
    model.delta = model.delta / v_mean
    logging.info("[k_norm] model.delta rescaled to %.6f", model.delta)
    # ===== [k_norm 끝] =====

    # setup optimizer
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
    logging.debug(log_params(model))
    optimizer = torch.optim.AdamW(
        params=[
            {"name": "encoding", "params": params_encoding},
            {"name": "net", "params": params_net, "weight_decay": 1e-2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    fp16 = not args.single_precision
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,
        enabled=fp16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        T_REG: args.weight_transformation,
        B_REG: args.weight_bias,
        I_REG: args.weight_image,
        D_REG: args.weight_deform,
        FF_LOSS: getattr(args, "weight_ff_loss", 0.0),
    }

    # ===== [FF Loss 추가] FF Loss 활성화 여부 및 패치 샘플링 하이퍼파라미터 =====
    use_ff_loss = model.ff_loss_fn is not None
    patch_size = getattr(args, "patch_size", 16)
    n_patches  = getattr(args, "n_patches",  8)

    # ===== [Hard Slice Mining] 슬라이스별 MSE 잔차 EMA 초기화 =====
    # [E3] 변경: torch.ones 초기화 → 첫 배치 실제 MSE로 초기화
    # 이유: 초기값 1.0이 실제 MSE(~1e-3~1e-1)보다 회등히 커서 EMA가
    #       수렴하는 데 ~100 iter가 소요되었음 (E2 워밍업의 근본 원인).
    #       첫 배치 MSE로 초기화하면 워밍업 없이도 즉시 올바른 잔차 분포를 사용.
    SLICE_RESIDUAL_EMA = 0.99
    n_slices = dataset.n_slices
    slice_residuals = torch.zeros(n_slices, dtype=torch.float32)   # CPU; 첫 배치 후 세팅
    slice_counts    = torch.zeros(n_slices, dtype=torch.long)      # 초기화 완료 여부 추적
    _residuals_initialized = False  # 첫 배치 초기화 완료 플래그
    logging.info("[E3] slice_residuals will be initialized from first batch MSE (no warmup needed).")
    # ===== [Hard Slice Mining 끝] =====

    # ===== [E2 호환성 유지] 워밍업 설정 (E3에서는 0으로 고정 권장) =====
    # E3는 실제 MSE 초기화로 워밍업 자체가 불필요하지만,
    # --hard_mining_warmup 인자를 여전히 지원하여 E1/E2 호환성을 유지한다.
    # E3 관접에서는 0으로 설정하면 첫 배치 이후 즉시 Hard Mining 활성화됨.
    HARD_MINING_WARMUP = getattr(args, "hard_mining_warmup", 0)
    logging.info(
        "[E3] Hard Slice Mining warmup: %d iters (0 = immediate activation after first-batch init).",
        HARD_MINING_WARMUP,
    )
    # ===== [E2 호환성 끝] =====

    if use_ff_loss:
        logging.info(
            "FF Loss enabled: weight=%.4f, patch_size=%d, n_patches=%d",
            loss_weights[FF_LOSS], patch_size, n_patches,
        )

    average = MovingAverage(1 - 0.001)
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()
        # forward
        batch = dataset.get_batch(args.batch_size, args.device)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)

            # ===== [Hard Slice Mining] SLICE_MSE_KEY 관리 =====
            if SLICE_MSE_KEY in losses:
                pixel_mse_vals, s_idx = losses.pop(SLICE_MSE_KEY)

                # ===== [E3] 첫 배치: EMA 대신 실제 MSE로 직접 초기화 =====
                if not _residuals_initialized:
                    for si in s_idx.unique():
                        mask = (s_idx == si)
                        slice_residuals[si.item()] = pixel_mse_vals[mask].mean().item()
                        slice_counts[si.item()] = 1
                    # 첫 배치에서 가싘되지 않은 슬라이스는 전체 평균으로 채우기
                    seen = slice_counts > 0
                    if seen.any():
                        init_mean = slice_residuals[seen].mean().item()
                        slice_residuals[~seen] = init_mean
                    _residuals_initialized = True
                    logging.info(
                        "[E3] slice_residuals initialized from first batch. "
                        "mean=%.6f, min=%.6f, max=%.6f",
                        slice_residuals.mean().item(),
                        slice_residuals.min().item(),
                        slice_residuals.max().item(),
                    )
                # ===== [E3 끝] =====

                # 모든 iter에서 EMA 업데이트 (초기화 후도 지속 추적)
                for si in s_idx.unique():
                    mask = (s_idx == si)
                    mean_mse = pixel_mse_vals[mask].mean().item()
                    slice_residuals[si.item()] = (
                        SLICE_RESIDUAL_EMA * slice_residuals[si.item()]
                        + (1 - SLICE_RESIDUAL_EMA) * mean_mse
                    )
            # ===== [Hard Slice Mining 끝] =====

            # ===== [FF Loss 추가] 패치 배치 샘플링 및 patch_forward 호출 =====
            if use_ff_loss:
                # i <= HARD_MINING_WARMUP: 균등 샘플링
                # i >  HARD_MINING_WARMUP: 잔차 EMA 기반 가중 샘플링 활성화
                # E3 기본값(HARD_MINING_WARMUP=0)에서는 첫 배치 초기화 후 즉시 활성화됨
                if i <= HARD_MINING_WARMUP or not _residuals_initialized:
                    sampling_probs = None
                else:
                    sampling_probs = slice_residuals / slice_residuals.sum()
                patch_batch = dataset.get_patch_batch(
                    n_patches, patch_size, args.device,
                    sampling_probs=sampling_probs,
                )
                if patch_batch:
                    patch_losses = model.patch_forward(**patch_batch)
                    losses.update(patch_losses)
            # ===== [FF Loss 추가 끝] =====

            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()
        if args.debug:
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_time += time.time() - train_step_start
        for k in losses:
            average(k, losses[k].item())
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            if not logging_header:
                train_logger = TrainLogger(
                    "time",
                    "epoch",
                    "iter",
                    *list(losses.keys()),
                    "lr",
                )
                logging_header = True
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                *[average[k] for k in losses],
                optimizer.param_groups[0]["lr"],
            )
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    logging.warning(
                        "Numerical instability detected! "
                        "The scale of GradScaler is %f, which is too small. "
                        "The results might be suboptimal. "
                        "Try to set --single-precision or run the command again with a different random seed."
                    )
                if i == args.n_iter:
                    logging.debug("Final scale of GradScaler = %f" % current_scaler)

    # outputs
    transformation = model.transformation
    ax = transformation.axisangle()
    ax[:, -3:] *= spatial_scaling
    transformation = RigidTransform(ax)
    transformation = RigidTransform(
        torch.cat([torch.zeros_like(center), center])[None]
    ).compose(transformation)
    model.inr.bounding_box.copy_(bb)
    dataset.xyz *= spatial_scaling

    dataset.transformation = transformation
    mask = dataset.mask
    output_slices = []
    for i in range(len(slices)):
        output_slice = slices[i].clone()
        output_slice.transformation = transformation[i]
        output_slices.append(output_slice)
    return model.inr, output_slices, mask

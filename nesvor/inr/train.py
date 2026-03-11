from argparse import Namespace
from typing import List, Tuple
import time
import datetime
import torch
import torch.optim as optim
import logging
from ..utils import MovingAverage, log_params, TrainLogger
# ===== [FF Loss 추가] 모델 임포트에 FF_LOSS 키 추가 =====
from .models import INR, NeSVoR, D_LOSS, S_LOSS, DS_LOSS, I_REG, B_REG, T_REG, D_REG, FF_LOSS
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
    # 주의: NeSVoR() 생성자 내부에서 dataset.mean으로 model.delta가 설정되므로
    # 반드시 NeSVoR 생성 이훈에 호출해야 함.
    v_mean = dataset.mean
    logging.info("[k_norm] Training target normalization: v_mean = %.4f", v_mean)
    dataset.normalize(v_mean)          # self.v 및 slice_images를 v_mean으로 나눔
    model.inr.v_mean.fill_(v_mean)     # 추론 시 역정규화에 쓸 인자를 INR 버퍼에 저장
    # model.delta는 NeSVoR 생성 시 args.delta * v_mean으로 설정되었음.
    # 정규화 훈 density 값이 ~1.0 수준이 되므로
    # delta도 v_mean으로 나누어 정규화된 공간에서의 임계값으로 복원해야 함.
    # (img_reg의 edge 모드: delta를 grad 임계값으로 사용하므로
    #  delta가 너무 크면 regularization이 사실상 비활성화됨)
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
    # logging
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
    # setup scheduler for lr decay
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    # setup grad scalar for mixed precision training
    fp16 = not args.single_precision
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,
        enabled=fp16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
    # training
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        T_REG: args.weight_transformation,
        B_REG: args.weight_bias,
        I_REG: args.weight_image,
        D_REG: args.weight_deform,
        # ===== [FF Loss 추가] loss_weights 딕셔너리에 FF_LOSS 가중치 등록 =====
        # args에 weight_ff_loss가 없으면 0.0으로 간주 (parsers.py 추가 전 호환성)
        FF_LOSS: getattr(args, "weight_ff_loss", 0.0),
        # ===== [FF Loss 추가 끝] =====
        # ===== [추가] 레벨 페널티 가중치 =====
        "levelReg": 0.05, # 초기값으로 0.05 정도를 추천합니다. 상황에 따라 파라미터로 빼셔도 좋습니다.
        # ===== [추가 끝] =====
    }

    # ===== [FF Loss 추가] FF Loss 활성화 여부 및 패치 샘플링 하이퍼파라미터 설정 =====
    # model.ff_loss_fn이 초기화된 경우(weight_ff_loss > 0)에만 패치 샘플링 수행
    use_ff_loss = model.ff_loss_fn is not None
    patch_size = getattr(args, "patch_size", 16)   # 패치 한 변 크기 (P×P)
    n_patches  = getattr(args, "n_patches",  8)    # iteration당 샘플링할 패치 수
    if use_ff_loss:
        logging.info(
            "FF Loss enabled: weight=%.4f, patch_size=%d, n_patches=%d",
            loss_weights[FF_LOSS], patch_size, n_patches,
        )
    # ===== [FF Loss 추가 끝] =====

    average = MovingAverage(1 - 0.001)
    # logging
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()
        # 가중치 관산용
        if i % 500 == 0:
            print("Learned Hash Grid Weights:", model.inr.level_weights.data)
        # forward
        batch = dataset.get_batch(args.batch_size, args.device)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)

            # ===== [FF Loss 추가] 패치 배치 샘플링 및 patch_forward 호출 =====
            # 픽셀 단위 MSE Loss와 동일한 backward()에서 함께 계산됨
            if use_ff_loss:
                patch_batch = dataset.get_patch_batch(
                    n_patches, patch_size, args.device
                )
                # 유효한 패치가 하나라도 있을 때만 FF Loss 추가
                if patch_batch:
                    patch_losses = model.patch_forward(**patch_batch)
                    losses.update(patch_losses)  # FF_LOSS 키를 losses에 합치
            # ===== [FF Loss 추가 끝] =====

            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()
        if args.debug:  # check nan grad
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
            # logging
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
            # check scaler
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

    # undo centering and scaling
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

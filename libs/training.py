"""Distributed data-loss-only training utilities for RB2D models."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _positional_list(x_coord, y_coord, device):
    x = x_coord[0] if x_coord.ndim > 1 else x_coord
    y = y_coord[0] if y_coord.ndim > 1 else y_coord
    return [x.to(device).unsqueeze(-1), y.to(device).unsqueeze(-1)]


def _predict(model, x_cur, pos_lst, model_type):
    if model_type == "fno":
        frame = x_cur[:, 0] if x_cur.ndim == 5 else x_cur
        out = model(frame.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1).contiguous()
    return model(x_cur, pos_lst)


def _step_input(x_cur, pred, in_time_window):
    pred_window = pred.unsqueeze(1)
    if in_time_window <= 1:
        return pred_window
    return torch.cat((x_cur[:, 1:], pred_window), dim=1)


def _mean_loss(model, x, y, xc, yc, model_type, steps, loss_fn, denorm, in_time_window):
    pos = _positional_list(xc, yc, x.device)
    x_cur = x
    total = x.new_zeros(())
    for step in range(steps):
        pred = _predict(model, x_cur, pos, model_type)
        target = y[:, step]
        x_cur = _step_input(x_cur, pred, in_time_window)
        if denorm is not None:
            pred = denorm(pred)
            target = denorm(target)
        total = total + torch.mean(loss_fn(pred, target))
    return total / steps


def _average(total, count, device):
    value = torch.tensor([total, count], dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float(value[0] / value[1]) if value[1] else 0.0


def train_one_epoch(model: DDP, train_loader: DataLoader, train_sampler: DistributedSampler,
                    optimizer, scaler, loss_fn, steps, model_type, device, epoch,
                    main_process, config, denorm=None, max_batches: Optional[int] = None,
                    **_kwargs):
    model.train(); train_sampler.set_epoch(epoch)
    total = 0.0; count = 0; updates = 0
    for index, (x, y, _t, xc, yc) in enumerate(train_loader):
        if max_batches is not None and index >= max_batches: break
        x, y, xc, yc = (v.to(device, non_blocking=True) for v in (x, y, xc, yc))
        with autocast(enabled=scaler is not None):
            loss = _mean_loss(model, x, y, xc, yc, model_type, steps, loss_fn,
                              denorm if config.data.normalize else None,
                              int(config.model.in_time_window))
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        total += float(loss.detach()) * x.size(0); count += x.size(0); updates += 1
        if main_process and index % 10 == 0:
            print(f"Epoch {epoch}, Iter {index}/{len(train_loader)}, data_loss: {loss.item():.6e}")
    return _average(total, count, device), updates


def evaluate(model: DDP, val_loader, val_sampler, loss_fn, use_autocast, steps,
             model_type, device, epoch, config, denorm=None, main_process=False,
             max_batches: Optional[int] = None, **_kwargs):
    if val_loader is None: return 0.0
    model.eval()
    if val_sampler is not None: val_sampler.set_epoch(epoch)
    total = 0.0; count = 0
    with torch.no_grad():
        for index, (x, y, _t, xc, yc) in enumerate(val_loader):
            if max_batches is not None and index >= max_batches: break
            x, y, xc, yc = (v.to(device, non_blocking=True) for v in (x, y, xc, yc))
            with autocast(enabled=use_autocast):
                loss = _mean_loss(model, x, y, xc, yc, model_type, steps, loss_fn,
                                  denorm if config.data.normalize else None,
                                  int(config.model.in_time_window))
            total += float(loss) * x.size(0); count += x.size(0)
    return _average(total, count, device)


def build_scheduler(optimizer, training):
    epochs = int(training.epochs)
    warmup_epochs = min(max(0, int(getattr(training, "warmup_epochs", 0))), max(0, epochs - 1))
    start_factor = float(getattr(training, "warmup_start_factor", 1e-3))
    eta_min = float(getattr(training, "min_lr", 1e-6))
    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)


def create_run_directory(config, model_type: str, seed: int) -> Path:
    root = Path(config.log_dir)
    if not root.is_absolute(): root = Path(__file__).resolve().parents[1] / root
    root.mkdir(parents=True, exist_ok=True)
    run = root / f"{model_type}_seed{seed}"
    index = 0
    while run.exists():
        index += 1; run = root / f"{model_type}_seed{seed}_{index}"
    run.mkdir(parents=True)
    return run

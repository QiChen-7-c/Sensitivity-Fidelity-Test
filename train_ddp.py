from models.factory import build_model
from libs.training import train_one_epoch, evaluate, build_scheduler, create_run_directory
from libs.configuration import apply_runtime_defaults, resolve_training_paths

import argparse
import os
from pathlib import Path
from timeit import default_timer
from typing import List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import yaml

from libs.distributed_utils import resolve_local_batch_size
from libs.rb_datasets import RB2D_Dataset
from libs.tools import (
    LpLoss,
    count_params,
    dict2namespace,
)


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Distributed training entry for IFactFormer (RB2D dataset)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/ifactformer.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model initialization and training order.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Random seed used only for the train/validation split.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="torch.distributed backend to use (e.g., nccl, gloo).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by torch.distributed launchers.",
    )
    parser.add_argument(
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers per process.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (in addition to best/final).",
    )
    parser.add_argument(
        "--epoch-one-step",
        type=int,
        default=None,
        help="Number of initial epochs trained with single-step data loss only.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_config_overrides(unknown_args: List[str]) -> dict:
    """
    Parse CLI overrides in dotted-key style, e.g.:
      --training.lr 1e-4
      --model.n_layer 4
      --model.some_list=[1,2,3]
    Values are parsed with yaml.safe_load for numeric/bool/list support.
    """
    overrides = {}
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith("--"):
            raise ValueError(f"Invalid override token '{token}'. Expected '--<key> <value>'.")

        key_expr = token[2:]
        if "=" in key_expr:
            key, raw_value = key_expr.split("=", 1)
            if not key:
                raise ValueError(f"Invalid override token '{token}'.")
            overrides[key] = yaml.safe_load(raw_value)
            i += 1
            continue

        key = key_expr
        if i + 1 >= len(unknown_args):
            raise ValueError(f"Missing value for override '{token}'.")
        raw_value = unknown_args[i + 1]
        if raw_value.startswith("--"):
            raise ValueError(f"Missing value for override '{token}'.")
        overrides[key] = yaml.safe_load(raw_value)
        i += 2

    return overrides


def apply_config_overrides(config_dict: dict, overrides: dict):
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor = config_dict

        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(
                    f"Unknown override path '{dotted_key}'. Failed at '{part}'."
                )
            cursor = cursor[part]

        leaf = parts[-1]
        if leaf not in cursor:
            raise KeyError(f"Unknown override key '{dotted_key}'.")
        cursor[leaf] = value


def load_config(config_path: Path, overrides: Optional[dict] = None):
    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)
    if overrides:
        apply_config_overrides(config_dict, overrides)
    resolve_training_paths(config_dict)
    return dict2namespace(config_dict)


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_distributed_mode(args: argparse.Namespace):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for train_ddp.py")

    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    env_local_rank = os.environ.get("LOCAL_RANK")

    if env_rank is None or env_world_size is None:
        raise RuntimeError(
            "RANK and WORLD_SIZE must be set. Launch with torchrun or torch.distributed.launch."
        )

    world_size = int(env_world_size)
    rank = int(env_rank)
    local_rank = args.local_rank if args.local_rank >= 0 else int(env_local_rank or 0)

    dist.init_process_group(backend=args.dist_backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def prepare_dataloaders(
    dataset: RB2D_Dataset,
    batch_size: int,
    val_ratio: float,
    split_seed: int,
    train_seed: int,
    num_workers: int,
    pin_memory: bool,
    train_subset_ratio: Optional[float] = None,
):
    split_generator = torch.Generator().manual_seed(split_seed)

    if train_subset_ratio is not None:
        subset_ratio = max(0.0, min(1.0, train_subset_ratio))
        if subset_ratio == 0.0:
            raise ValueError("train_subset_ratio cannot be 0; no samples left for training.")
        subset_size = max(1, int(len(dataset) * subset_ratio)) # choose small subset for quick debugging
        subset_indices = torch.randperm(
            len(dataset), generator=split_generator
        )[:subset_size].tolist()
        dataset = Subset(dataset, subset_indices)

    val_ratio = max(0.0, min(1.0, val_ratio))
    n_samples = len(dataset)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val
    if n_train == 0:
        raise ValueError("Validation ratio too large; no samples left for training.")

    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=split_generator
    )

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        seed=train_seed,
        drop_last=False,
    )
    val_sampler = None
    if n_val > 0:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    def _make_loader(subset, sampler):
        return DataLoader(
            subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    train_loader = _make_loader(train_dataset, train_sampler)
    val_loader = _make_loader(val_dataset, val_sampler) if n_val > 0 else None

    return train_loader, val_loader, train_sampler, val_sampler


def broadcast_run_directory(run_dir: Path) -> Path:
    run_dir_str = str(run_dir)
    object_list = [run_dir_str]
    dist.broadcast_object_list(object_list, src=0)
    return Path(object_list[0])


def write_log_header(log_file: Path, args_snapshot, config_text: Optional[str], param_info: str):
    with log_file.open("w") as fh:
        fh.write("# Arguments\n")
        for key, value in args_snapshot.items():
            fh.write(f"{key}: {value}\n")
        if config_text:
            fh.write("\n# Config\n")
            fh.write(config_text.rstrip() + "\n")
        fh.write("\n" + param_info + "\n")


def main():
    args, unknown_args = parse_args()
    overrides = parse_config_overrides(unknown_args)
    config = load_config(args.config, overrides=overrides)
    apply_runtime_defaults(args, config)

    if args.dist_backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires at least one CUDA device.")

    rank, world_size, local_rank, device = init_distributed_mode(args)
    main_process = is_main_process(rank)
    global_batch_size = int(config.training.batch_size)
    per_rank_batch_size = resolve_local_batch_size(global_batch_size, world_size)

    args_snapshot = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    args_snapshot.update(
        {
            "world_size": world_size,
            "global_rank": rank,
            "effective_local_rank": local_rank,
            "device": str(device),
            "global_batch_size": global_batch_size,
            "per_rank_batch_size": per_rank_batch_size,
        }
    )
    if overrides:
        args_snapshot["cli_overrides"] = overrides

    config_text_for_log = None
    config_for_checkpoint = None
    if main_process:
        try:
            with args.config.open("r") as f:
                config_for_checkpoint = yaml.safe_load(f)
            if overrides:
                apply_config_overrides(config_for_checkpoint, overrides)
            resolve_training_paths(config_for_checkpoint)
            for key in ("seed", "split_seed", "val_ratio", "num_workers", "save_every", "epoch_one_step"):
                config_for_checkpoint["training"][key] = getattr(args, key)
            config_text_for_log = yaml.safe_dump(
                config_for_checkpoint, sort_keys=False, default_flow_style=False
            )
        except OSError:
            config_text_for_log = None
            config_for_checkpoint = None

    set_random_seed(args.seed + rank)


    dataset = RB2D_Dataset(
        data_folder=config.data.data_folder,
        data_filenames=config.data.train_data_files,
        nx=config.data.nx,
        ny=config.data.ny,
        nt_in=config.model.in_time_window,
        nt_out=config.model.out_time_window,
        stride_t=config.data.stride,
        normalize_channels=config.data.normalize,
        subsample_rate=config.data.subsample_rate,
        BC=False,
    )

    denorm = dataset.denormalize_grid

    train_loader, val_loader, train_sampler, val_sampler = prepare_dataloaders(
        dataset=dataset,
        batch_size=per_rank_batch_size,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        train_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        # train_subset_ratio=0.01, # for quick debugging
    )

    if main_process:
        run_dir = create_run_directory(config, "ifactformer", args.seed)
    else:
        run_dir = Path(".")
    run_dir = broadcast_run_directory(run_dir)
    dist.barrier()

    log_file = run_dir / "training_epoch_log.txt"
    model = build_model(config, "ifactformer").to(device)
    param_info = f"count_params: {count_params(model)}"

    if main_process:
        print(run_dir)
        print(
            f"Global batch size: {global_batch_size} | "
            f"Per-rank batch size: {per_rank_batch_size}"
        )
        print(param_info)
        write_log_header(log_file, args_snapshot, config_text_for_log, param_info)

    model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler = build_scheduler(optimizer, config.training)

    loss_fn = LpLoss(reduction=False)
    scaler = GradScaler() if device.type == "cuda" else None
    use_autocast = scaler is not None

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    best_val_loss = float("inf")

    total_epochs = config.training.epochs
    steps = config.model.out_time_window
    if main_process:
        print("\n" + "=" * 50)
        print("Starting distributed training...")
        print("=" * 50 + "\n")

    t0 = default_timer()
    try:
        for epoch in range(total_epochs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device=device)

            use_one_step = epoch < args.epoch_one_step
            train_steps = 1 if use_one_step else steps
            eval_steps = steps

            train_loss, optimizer_steps = train_one_epoch(
                model=model,
                train_loader=train_loader,
                train_sampler=train_sampler,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                steps=train_steps,
                model_type="ifactformer",
                device=device,
                epoch=epoch,
                main_process=main_process,
                log_interval=10,
                config=config,
                denorm=denorm,
                max_batches=args.max_train_batches,
            )

            val_loss = evaluate(
                model=model,
                val_loader=val_loader,
                val_sampler=val_sampler,
                loss_fn=loss_fn,
                use_autocast=use_autocast,
                steps=eval_steps,
                model_type="ifactformer",
                device=device,
                epoch=epoch,
                config=config,
                denorm=denorm,
                main_process=main_process,
                max_batches=args.max_val_batches,
            )

            if optimizer_steps > 0:
                scheduler.step()
            elif main_process:
                warnings.warn(
                    "Skip lr scheduler step because optimizer.step() was not called this epoch."
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
                peak_mem_usage = distributed_max(
                    float(torch.cuda.max_memory_allocated(device=device)),
                    device,
                ) / 1024**3
            else:
                peak_mem_usage = 0.0

            if main_process:
                train_loss_hist.append(train_loss)
                val_loss_hist.append(val_loss)

                elapsed = default_timer() - t0
                lr = optimizer.param_groups[0]["lr"]

                log_line = (
                    f"Epoch {epoch:3d} | Time: {elapsed:.2f}s | "
                    f"train_loss: {train_loss:.8f} | val_loss: {val_loss:.8f} | LR: {lr:.8g}"
                )
                print(log_line)
                with log_file.open("a") as fh:
                    fh.write(log_line + "\n")

                loss_log = {
                    "train_loss": np.array(train_loss_hist, dtype=np.float32),
                    "val_loss": np.array(val_loss_hist, dtype=np.float32),
                }
                np.save(run_dir / "loss_log.npy", loss_log, allow_pickle=True)

                model_state = model.module.state_dict()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {"model_state": model_state, "config": config_for_checkpoint},
                        run_dir / "checkpoint_best.pt",
                    )
                    print(f"  >>> Saved best model with test L2: {val_loss:.6f}")

                if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                    torch.save(
                        {"model_state": model.module.state_dict(), "config": config_for_checkpoint},
                        run_dir / f"checkpoint_{epoch + 1}.pt",
                    )

            dist.barrier()

        if main_process:
            model_state = model.module.state_dict()
            torch.save(
                {"model_state": model_state, "config": config_for_checkpoint},
                run_dir / "checkpoint_final.pt",
            )
            print("\n" + "=" * 50)
            print(f"Training completed in {default_timer() - t0:.2f}s")
            print(f"Best test L2: {best_val_loss:.6f}")
            print(f"Results saved to: {run_dir}")
            print("=" * 50)

        dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

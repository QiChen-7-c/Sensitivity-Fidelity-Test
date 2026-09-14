"""Small configuration helpers shared by the four RB training entries."""
from __future__ import annotations
from pathlib import Path


def resolve_training_paths(config_dict: dict) -> dict:
    data = config_dict.get("data", {})
    folder = data.get("data_folder")
    if folder:
        path = Path(folder)
        if not path.is_absolute():
            data["data_folder"] = str((Path(__file__).resolve().parents[1] / path).resolve())
    return config_dict


def apply_runtime_defaults(args, config) -> None:
    training = config.training
    if args.seed is None:
        args.seed = int(getattr(training, "seed", 1234))
    if args.split_seed is None:
        args.split_seed = int(getattr(training, "split_seed", args.seed))
    if args.val_ratio is None:
        args.val_ratio = float(getattr(training, "val_ratio", 0.1))
    if args.num_workers is None:
        args.num_workers = int(getattr(training, "num_workers", 4))
    if getattr(args, "save_every", None) is None:
        args.save_every = int(getattr(training, "save_every", 5))
    if getattr(args, "epoch_one_step", None) is None:
        args.epoch_one_step = int(getattr(training, "epoch_one_step", 0))

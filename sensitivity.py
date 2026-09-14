import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch
import yaml

from models.factory import build_model, infer_model_type
from libs.tools import count_params, dict2namespace
from libs.rb_datasets import RB2D_Dataset


def parse_args() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(
        description="Sensitivity study for IFactFormer / AFNO / FNO / Swin multi-step models."
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        default=Path("outputs/ifactformer"),
        help="Directory that stores checkpoint .pt with embedded config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoint_best.pt',
        help="Checkpoint filename inside load_dir. If omitted, tries best/final checkpoints automatically.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "ifactformer", "afno", "fno", "swin"],
        help="Model family. 'auto' infers from checkpoint config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g., cuda:0 or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset index to analyse.",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help="Zero-based prediction step to use as sensitivity target. Default: final step.",
    )
    parser.add_argument(
        "--out-time-window",
        type=int,
        default=None,
        help="Override config.model.out_time_window for dataset horizon and rollout steps.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=8,
        help="Square ROI size (grid cells) used for the target functional.",
    )
    parser.add_argument(
        "--roi-channel",
        type=int,
        default=0,
        help="Channel index used for the target functional (0=B, 1=U, 2=V).",
    )
    parser.add_argument(
        "--roi-x-start",
        type=int,
        default=None,
        help="Optional ROI start index along x (overrides automatic centering).",
    )
    parser.add_argument(
        "--roi-y-start",
        type=int,
        default=None,
        help="Optional ROI start index along y (overrides automatic centering).",
    )
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=1,
        help="Visualization interval in steps (plot one frame every N steps).",
    )
    parser.add_argument(
        "--physical-io",
        action="store_true",
        help=(
            "Use physical (denormalized) input/output tensors in the sensitivity "
            "graph while keeping the model rollout in normalized space. "
            "Equivalent to the former sensitivity_test.py behavior."
        ),
    )
    parser.add_argument(
        "--adjoint-path",
        type=Path,
        default=Path("./data/adjoint_1e-2_1000_r5e4.npz"),
        help=(
            "Path to adjoint NPZ used for gradient comparison. If the default "
            "relative path is missing, the script also tries "
            "data/<filename>."
        ),
    )
    args, unknown = parser.parse_known_args()
    overrides = parse_config_overrides(unknown)
    return args, overrides


def parse_config_overrides(unknown_args: List[str]) -> dict:
    """
    Parse CLI overrides in dotted-key style, e.g.:
      --data.test_data_files "['test_data.npy']"
      --model.out_time_window 20
      --training.batch_size 8
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


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_and_config(
    load_dir: Path,
    checkpoint_name: Optional[str],
    overrides=None,
):
    checkpoint_path = resolve_checkpoint_path(load_dir, checkpoint_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # PyTorch 2.6 defaults `weights_only=True`, which can fail for full training
    # checkpoints that contain non-tensor objects (e.g., config / metadata).
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept `weights_only`.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    config_dict = checkpoint.get("config")
    if config_dict is None:
        raise ValueError("Config not found in checkpoint.")
    if overrides:
        apply_config_overrides(config_dict, overrides)
    config = dict2namespace(config_dict)
    state_dict = checkpoint.get("model_state", checkpoint)
    return state_dict, config, checkpoint_path


def resolve_checkpoint_path(load_dir: Path, checkpoint_name: Optional[str]) -> Path:
    if checkpoint_name:
        return load_dir / checkpoint_name

    # Prefer finetuned checkpoints when present, then generic best/final names.
    patterns = [
        "finetune_checkpoint_best*.pt",
        "checkpoint_best.pt",
        "finetune_checkpoint_final*.pt",
        "checkpoint_final.pt",
        "checkpoint_best.pth",
        "checkpoint_final.pth",
        "*.pt",
        "*.pth",
    ]
    for pattern in patterns:
        if "*" in pattern:
            matches = sorted(load_dir.glob(pattern))
            if matches:
                return matches[0]
        else:
            candidate = load_dir / pattern
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"No checkpoint found in {load_dir}. Pass --checkpoint explicitly."
    )


# Model construction is centralized in models.factory.

def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    original_error = None
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        original_error = exc

    has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())
    if not has_module_prefix:
        if original_error is not None:
            raise original_error
        raise RuntimeError("Failed to load state dict and no module.* prefix found.")

    stripped_state_dict = {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(stripped_state_dict)


def to_channels_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def to_channels_last(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def rollout_step(
    model: torch.nn.Module,
    x_cur: torch.Tensor,
    pos_lst: List[torch.Tensor],
    model_type: str,
    in_time_window: int,
) -> torch.Tensor:
    if model_type == "fno":
        if in_time_window != 1:
            raise ValueError(
                f"FNO sensitivity currently expects in_time_window=1, got {in_time_window}."
            )
        if x_cur.dim() == 5:
            x_frame = x_cur[:, 0]
        elif x_cur.dim() == 4:
            x_frame = x_cur
        else:
            raise ValueError(f"Unexpected FNO input shape: {tuple(x_cur.shape)}")
        y_pred = model(to_channels_first(x_frame))
        if y_pred.dim() != 4:
            raise ValueError(f"Unexpected FNO output shape: {tuple(y_pred.shape)}")
        return to_channels_last(y_pred)

    y_pred = model(x_cur, pos_lst)
    if y_pred.dim() == 5:
        if y_pred.size(1) != 1:
            raise ValueError(
                f"Unexpected 5D model output shape: {tuple(y_pred.shape)}"
            )
        y_pred = y_pred[:, 0]
    if y_pred.dim() != 4:
        raise ValueError(f"Unexpected model output shape: {tuple(y_pred.shape)}")
    return y_pred


def update_autoregressive_input(
    x_cur: torch.Tensor, y_pred_step: torch.Tensor, in_time_window: int
) -> torch.Tensor:
    if in_time_window <= 1:
        return y_pred_step

    if x_cur.dim() != 5:
        raise ValueError(
            f"Expected current input in [B,T,X,Y,C], got shape {tuple(x_cur.shape)}"
        )
    if y_pred_step.dim() != 5 or y_pred_step.size(1) != 1:
        raise ValueError(
            f"Expected predicted step in [B,1,X,Y,C], got shape {tuple(y_pred_step.shape)}"
        )
    if x_cur.size(1) != in_time_window:
        raise ValueError(
            f"Expected input time window {in_time_window}, got {x_cur.size(1)}."
        )
    return torch.cat([x_cur[:, 1:, ...], y_pred_step], dim=1)


def select_data_files(config) -> List[str]:
    for key in ("valid_data_files", "test_data_files", "val_data_files", "train_data_files"):
        data_files = getattr(config.data, key, None)
        if data_files:
            return data_files
    raise ValueError("No data files found in config (expected valid/test/val/train entries).")


def select_normalization_files(config) -> List[str]:
    data_files = getattr(config.data, "train_data_files", None)
    if data_files:
        return data_files
    return select_data_files(config)


def build_pos_lst_from_coords(
    x_coord: torch.Tensor, y_coord: torch.Tensor, device: torch.device
) -> List[torch.Tensor]:
    pos_x = x_coord[0] if x_coord.dim() > 1 else x_coord
    pos_y = y_coord[0] if y_coord.dim() > 1 else y_coord
    pos_x = pos_x.to(device, non_blocking=True).unsqueeze(-1)
    pos_y = pos_y.to(device, non_blocking=True).unsqueeze(-1)
    return [pos_x, pos_y]


def compute_roi_bounds(
    nx: int,
    ny: int,
    roi_size: int,
    roi_x_start: Optional[int] = None,
    roi_y_start: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    if roi_x_start is None:
        roi_x_start = max(0, (nx - roi_size) // 2)
    if roi_y_start is None:
        roi_y_start = max(0, (ny - roi_size) // 2)
    roi_x_end = roi_x_start + roi_size
    roi_y_end = roi_y_start + roi_size
    if roi_x_end > nx or roi_y_end > ny:
        raise ValueError("ROI exceeds spatial domain, reduce roi_size.")
    return roi_x_start, roi_x_end, roi_y_start, roi_y_end


def infer_output_prefix(checkpoint_path: Path) -> str:
    checkpoint_name = checkpoint_path.stem.lower()
    if checkpoint_name.startswith("finetune"):
        return "finetune"
    return "pretrain"


def sanitize_path_component(value: str) -> str:
    cleaned = [
        ch if ch.isalnum() or ch in ("-", "_", ".") else "-"
        for ch in value.strip()
    ]
    text = "".join(cleaned).strip("-")
    return text or "analysis"


def build_analysis_name(
    model_type: str,
    load_dir: Path,
    checkpoint_path: Path,
    sample_index: int,
    target_step: int,
    physical_io: bool,
) -> str:
    mode_tag = "physical" if physical_io else "normalized"
    checkpoint_tag = infer_output_prefix(checkpoint_path)
    base_name = (
        f"sensitivity_{model_type}_{checkpoint_tag}_{load_dir.name}_"
        f"sample{sample_index}_target{target_step + 1}_{mode_tag}"
    )
    return sanitize_path_component(
        f"box2_{base_name}"
    )


def create_unique_output_dir(results_root: Path, analysis_name: str) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    output_dir = results_root / analysis_name
    suffix = 0
    while output_dir.exists():
        suffix += 1
        output_dir = results_root / f"{analysis_name}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_analysis_artifacts(
    output_dir: Path,
    output_prefix: str,
    input_states_physical: np.ndarray,
    ground_truth_future_physical: np.ndarray,
    prediction_future_physical: np.ndarray,
    ground_truth_full_physical: np.ndarray,
    prediction_full_physical: np.ndarray,
    sensitivity: np.ndarray,
    sensitivity_space: str,
    input_times: np.ndarray,
    pred_times: np.ndarray,
    state_times: np.ndarray,
    sensitivity_times: np.ndarray,
    model_x_coords: np.ndarray,
    model_y_coords: np.ndarray,
    roi_bounds_index: Tuple[int, int, int, int],
    roi_bounds_physical: Tuple[float, float, float, float],
    state_step_labels: List[str],
    sensitivity_step_labels: List[str],
    data_files: List[str],
    normalization_files: Optional[List[str]],
    model_type: str,
    checkpoint_path: Path,
    load_dir: Path,
    sample_index: int,
    target_step: int,
    roi_channel: int,
    physical_io: bool,
) -> None:
    np.save(
        output_dir / f"{output_prefix}_input_physical.npy",
        np.asarray(input_states_physical, dtype=np.float32),
    )
    np.save(
        output_dir / f"{output_prefix}_ground_truth_future_physical.npy",
        np.asarray(ground_truth_future_physical, dtype=np.float32),
    )
    np.save(
        output_dir / f"{output_prefix}_prediction_future_physical.npy",
        np.asarray(prediction_future_physical, dtype=np.float32),
    )
    np.save(
        output_dir / f"{output_prefix}_ground_truth_full_physical.npy",
        np.asarray(ground_truth_full_physical, dtype=np.float32),
    )
    np.save(
        output_dir / f"{output_prefix}_prediction_full_physical.npy",
        np.asarray(prediction_full_physical, dtype=np.float32),
    )
    np.save(
        output_dir / f"{output_prefix}_sensitivity_{sensitivity_space}.npy",
        np.asarray(sensitivity, dtype=np.float32),
    )
    np.savez_compressed(
        output_dir / f"{output_prefix}_coords_times.npz",
        input_times=np.asarray(input_times, dtype=np.float64),
        pred_times=np.asarray(pred_times, dtype=np.float64),
        state_times=np.asarray(state_times, dtype=np.float64),
        sensitivity_times=np.asarray(sensitivity_times, dtype=np.float64),
        model_x_coords=np.asarray(model_x_coords, dtype=np.float64),
        model_y_coords=np.asarray(model_y_coords, dtype=np.float64),
        roi_bounds_index=np.asarray(roi_bounds_index, dtype=np.int64),
        roi_bounds_physical=np.asarray(roi_bounds_physical, dtype=np.float64),
    )

    metadata = {
        "model_type": model_type,
        "load_dir": str(load_dir),
        "checkpoint_path": str(checkpoint_path),
        "sample_index": int(sample_index),
        "target_step": int(target_step),
        "roi_channel": int(roi_channel),
        "physical_io": bool(physical_io),
        "sensitivity_space": sensitivity_space,
        "data_files": list(data_files),
        "normalization_files": list(normalization_files) if normalization_files is not None else None,
        "state_step_labels": list(state_step_labels),
        "sensitivity_step_labels": list(sensitivity_step_labels),
    }
    with (output_dir / f"{output_prefix}_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)


def build_rollout_times(
    dataset: RB2D_Dataset,
    sample_index: int,
    nt_in: int,
    predict_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, t_id, _, _ = dataset.rand_start_id[sample_index]
    t_id = int(t_id)
    all_times = np.asarray(dataset.t_coord.detach().cpu().numpy(), dtype=np.float64)

    input_start = t_id
    input_end = input_start + nt_in
    pred_start = t_id + dataset.stride_t
    pred_end = pred_start + predict_steps
    if input_end > all_times.shape[0] or pred_end > all_times.shape[0]:
        raise ValueError(
            "Time coordinates do not cover the requested visualization window: "
            f"input [{input_start}:{input_end}), pred [{pred_start}:{pred_end}), "
            f"available={all_times.shape[0]}."
        )

    input_times = all_times[input_start:input_end]
    pred_times = all_times[pred_start:pred_end]
    combined_times = np.concatenate([input_times, pred_times], axis=0)
    return input_times, pred_times, combined_times


def build_rollout_labels(
    nt_in: int,
    predict_steps: int,
    target_step: Optional[int] = None,
) -> List[str]:
    labels = [f"Input Step {idx + 1}" for idx in range(nt_in)]
    for step_idx in range(predict_steps):
        label = f"Pred Step {step_idx + 1}"
        if target_step is not None and step_idx == target_step:
            label += " (target)"
        labels.append(label)
    return labels


def cheb_nodes(n: int, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    if n < 2:
        return np.asarray([a], dtype=np.float64)
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    return (b - a) * (x + 1.0) / 2.0 + a


def build_adjoint_coords(nx: int, nz: int, lx: float = 3.0, lz: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, lx, nx, endpoint=False, dtype=np.float64)
    z = cheb_nodes(nz, 0.0, lz)
    return x, z


def clip_vector_magnitude(u: np.ndarray, v: np.ndarray, max_magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
    if max_magnitude <= 0:
        return np.zeros_like(u), np.zeros_like(v)
    magnitude = np.sqrt(u**2 + v**2)
    scale = np.ones_like(magnitude)
    mask = magnitude > max_magnitude
    scale[mask] = max_magnitude / (magnitude[mask] + 1e-12)
    return u * scale, v * scale


def roi_bounds_from_coords(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    roi_x_start: int,
    roi_x_end: int,
    roi_y_start: int,
    roi_y_end: int,
) -> Tuple[float, float, float, float]:
    if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
        raise ValueError("ROI bounds must have positive size.")
    return (
        float(x_coords[roi_x_start]),
        float(x_coords[roi_x_end - 1]),
        float(y_coords[roi_y_start]),
        float(y_coords[roi_y_end - 1]),
    )


def flip_roi_bounds_vertical(
    roi_bounds: Tuple[float, float, float, float],
    y_coords: np.ndarray,
) -> Tuple[float, float, float, float]:
    x0, x1, y0, y1 = [float(v) for v in roi_bounds]
    y_min = float(np.min(y_coords))
    y_max = float(np.max(y_coords))
    y_low, y_high = sorted((y0, y1))
    return x0, x1, y_min + y_max - y_high, y_min + y_max - y_low


def flip_roi_indices_vertical(
    roi_bounds: Tuple[int, int, int, int],
    ny: int,
) -> Tuple[int, int, int, int]:
    x0, x1, y0, y1 = [int(v) for v in roi_bounds]
    return x0, x1, ny - y1, ny - y0


def resolve_adjoint_path(adjoint_path: Path) -> Path:
    """Resolve an adjoint file from the working directory or this project."""
    if adjoint_path.is_absolute() and adjoint_path.exists():
        return adjoint_path
    candidates = [Path.cwd() / adjoint_path, Path(__file__).resolve().parent / adjoint_path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    candidate_text = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"Adjoint NPZ not found. Tried:\n{candidate_text}")

def load_adjoint_gradient_data(
    adjoint_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[Tuple[float, float, float, float]], Path]:
    resolved_path = resolve_adjoint_path(adjoint_path)
    payload = np.load(resolved_path)
    required_keys = ["grad_b", "grad_u_x", "grad_u_z", "times", "steps"]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise KeyError(
            f"Adjoint NPZ {resolved_path} is missing required keys: {missing_keys}"
        )

    grad_data = np.stack(
        [payload["grad_b"], payload["grad_u_x"], payload["grad_u_z"]],
        axis=-1,
    ).astype(np.float32, copy=False)
    grad_times = np.asarray(payload["times"], dtype=np.float64)
    grad_steps = np.asarray(payload["steps"])
    step_labels = [f"Adjoint Step {int(step)}" for step in grad_steps.tolist()]
    roi_bounds = None
    if "roi_bounds" in payload:
        roi_bounds_arr = np.asarray(payload["roi_bounds"], dtype=np.float64)
        if roi_bounds_arr.shape == (4,):
            roi_bounds = tuple(float(v) for v in roi_bounds_arr.tolist())
    return grad_data, grad_times, step_labels, roi_bounds, resolved_path


def plot_grad_pred(
    data,
    nx,
    ny,
    step_labels,
    times,
    dir,
    output_prefix,
    roi_bounds,
    type='grad',
    ind=0,
    viz_interval=1,
    temp_absmax_override: Optional[float] = None,
    invert_vertical: bool = False,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    transpose_fields: bool = True,
    roi_bounds_are_indices: bool = True,
):
    if viz_interval <= 0:
        raise ValueError(f"viz_interval must be > 0, got {viz_interval}.")

    temp = data[..., 0]
    u = data[..., 1]
    v = data[..., 2]

    if temp.shape[0] == 0:
        return

    sampled_indices = list(range(0, temp.shape[0], viz_interval))
    if not sampled_indices:
        return

    if temp_absmax_override is not None:
        temp_absmax = float(temp_absmax_override)
    else:
        temp_absmax = float(np.abs(temp).max())
    if temp_absmax <= 0:
        temp_absmax = 1e-8

    uv_magnitude = np.sqrt(u**2 + v**2)
    uv_absmax = float(uv_magnitude.max())
    if uv_absmax <= 0:
        uv_absmax = 1e-8
    quiver_scale = uv_absmax / 3.0
    # quiver_scale = 5e-3


    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)
    if type == 'adjoint' and invert_vertical:
        if roi_bounds_are_indices:
            roi_bounds = flip_roi_indices_vertical(roi_bounds, ny)
        else:
            roi_bounds = flip_roi_bounds_vertical(roi_bounds, y_coords)
    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_bounds
    roi_width = roi_x_end - roi_x_start
    roi_height = roi_y_end - roi_y_start

    mesh_indexing = "xy" if transpose_fields else "ij"
    X, Y = np.meshgrid(x_coords, y_coords, indexing=mesh_indexing)

    stride_y_grad = 4
    stride_x_grad = 4
    stride_y = 4
    stride_x = 4
    rows, cols = 4, 5
    plots_per_figure = rows * cols

    for fig_idx, start in enumerate(range(0, len(sampled_indices), plots_per_figure), start=1):
        figure_indices = sampled_indices[start : start + plots_per_figure]
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(21, 8),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        contour_ref = None

        for panel_idx, ax in enumerate(axes):
            if panel_idx >= len(figure_indices):
                ax.axis("off")
                continue

            step_idx = figure_indices[panel_idx]
            if transpose_fields:
                temp_field = temp[step_idx].T
                u_field = u[step_idx].T
                v_field = v[step_idx].T
            else:
                temp_field = temp[step_idx]
                u_field = u[step_idx]
                v_field = v[step_idx]
            norm = TwoSlopeNorm(vmin=-temp_absmax, vcenter=0.0, vmax=temp_absmax)
            if type in {'grad', 'adjoint'}:
                contour_ref = ax.contourf(
                    X,
                    Y,
                    temp_field,
                    levels=np.linspace(-temp_absmax, temp_absmax, 20),
                    cmap="coolwarm",
                    norm=norm,
                )
                if type == 'adjoint':
                    u_field, v_field = clip_vector_magnitude(u_field, v_field, uv_absmax)
                ax.quiver(
                    X[::stride_y_grad, ::stride_x_grad],
                    Y[::stride_y_grad, ::stride_x_grad],
                    u_field[::stride_y_grad, ::stride_x_grad],
                    v_field[::stride_y_grad, ::stride_x_grad],
                    # angles="xy",
                    # scale_units="xy",
                    # scale=1e-4,
                    color="k",
                    scale=8e-2 if type == 'adjoint' else 5e-2,
                    width=0.003,
                )
            else:
                contour_ref = ax.contourf(
                    X,
                    Y,
                    temp_field,
                    levels=20,
                    cmap="coolwarm",
                )
                ax.quiver(
                    X[::stride_y, ::stride_x],
                    Y[::stride_y, ::stride_x],
                    u_field[::stride_y, ::stride_x],
                    v_field[::stride_y, ::stride_x],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale * 0.5,
                    width=0.003,
                )

            roi_rect = Rectangle(
                (
                    roi_x_start - 0.5 if roi_bounds_are_indices else roi_x_start,
                    roi_y_start - 0.5 if roi_bounds_are_indices else roi_y_start,
                ),
                roi_width,
                roi_height,
                linewidth=1.5,
                edgecolor="orange" if type == 'adjoint' and not roi_bounds_are_indices else "black",
                facecolor="none",
            )
            ax.add_patch(roi_rect)

            if step_idx < len(step_labels):
                title = step_labels[step_idx]
            else:
                title = f"Step {step_idx + 1}"
            if times is not None and step_idx < len(times):
                title += f"\nt={times[step_idx]:.3f}"
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        if invert_vertical and axes.size > 0:
            # Match the adjoint plotting reference --invert-z.
            axes[0].invert_yaxis()

        if contour_ref is not None:
            cbar = fig.colorbar(contour_ref, ax=list(axes), fraction=0.04, pad=0.02)
            if type == 'grad':
                cbar.set_label("Gradient of T")
            elif type == 'adjoint':
                cbar.set_label("Adjoint Gradient of T")
            elif type == 'true':
                cbar.set_label("True T")
            else:
                cbar.set_label("Predicted T")

        if type == 'grad':
            figure_path = dir / f"{output_prefix}_gradient_interval{viz_interval}_part{fig_idx}_{ind}.png"
        elif type == 'adjoint':
            figure_path = dir / f"{output_prefix}_adjoint_gradient_interval{viz_interval}_part{fig_idx}_{ind}.png"
        elif type == 'true':
            figure_path = dir / f"{output_prefix}_true_interval{viz_interval}_part{fig_idx}_{ind}.png"
        else:
            figure_path = dir / f"{output_prefix}_predicted_interval{viz_interval}_part{fig_idx}_{ind}.png"
        fig.savefig(figure_path, dpi=1000)
        plt.close(fig)


def plot_pred_true_compare(
    pred_data,
    label_data,
    nx,
    ny,
    state_times,
    output_dir,
    output_prefix,
    ind=0,
    steps_per_figure=10,
):
    temp_pred = pred_data[..., 0]
    u_pred = pred_data[..., 1]
    v_pred = pred_data[..., 2]

    temp_label = label_data[..., 0]
    u_label = label_data[..., 1]
    v_label = label_data[..., 2]

    total_steps = min(temp_label.shape[0], temp_pred.shape[0])
    if total_steps == 0:
        return

    temp_min = float(min(temp_label[:total_steps].min(), temp_pred[:total_steps].min()))
    temp_max = float(max(temp_label[:total_steps].max(), temp_pred[:total_steps].max()))
    if np.isclose(temp_min, temp_max):
        delta = 1e-8 if temp_min == 0 else abs(temp_min) * 1e-6
        temp_min -= delta
        temp_max += delta
    temp_levels = np.linspace(temp_min, temp_max, 20)

    uv_label_mag = np.sqrt(u_label[:total_steps] ** 2 + v_label[:total_steps] ** 2)
    uv_pred_mag = np.sqrt(u_pred[:total_steps] ** 2 + v_pred[:total_steps] ** 2)
    uv_absmax = float(max(uv_label_mag.max(), uv_pred_mag.max()))
    if uv_absmax <= 0:
        uv_absmax = 1e-8
    quiver_scale = uv_absmax / 8.0

    x_coords = np.arange(nx)
    y_coords = np.arange(ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    stride_y = 4
    stride_x = 4

    for fig_idx, start_offset in enumerate(range(0, total_steps, steps_per_figure), start=1):
        time_steps = min(steps_per_figure, total_steps - start_offset)
        fig, axes = plt.subplots(
            2,
            steps_per_figure,
            figsize=(30, 8),
            constrained_layout=True,
            squeeze=False,
        )
        contour_ref = None

        for step in range(time_steps):
            ax_label = axes[0, step]
            ax_pred = axes[1, step]
            num_step = start_offset + step

            contour_ref = ax_label.contourf(
                X,
                Y,
                temp_label[num_step].T,
                levels=temp_levels,
                cmap="coolwarm",
            )
            ax_label.quiver(
                X[::stride_y, ::stride_x],
                Y[::stride_y, ::stride_x],
                u_label[num_step].T[::stride_y, ::stride_x],
                v_label[num_step].T[::stride_y, ::stride_x],
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=0.003,
            )
            label_title = f"Label Step {num_step + 1}"
            if num_step < len(state_times):
                label_title += f"\nt={state_times[num_step]:.3f}"
            ax_label.set_title(label_title, fontsize=10)
            ax_label.set_xticks([])
            ax_label.set_yticks([])
            ax_label.set_aspect("equal")

            ax_pred.contourf(
                X,
                Y,
                temp_pred[num_step].T,
                levels=temp_levels,
                cmap="coolwarm",
            )
            ax_pred.quiver(
                X[::stride_y, ::stride_x],
                Y[::stride_y, ::stride_x],
                u_pred[num_step].T[::stride_y, ::stride_x],
                v_pred[num_step].T[::stride_y, ::stride_x],
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=0.003,
            )
            pred_title = f"Prediction Step {num_step + 1}"
            if num_step < len(state_times):
                pred_title += f"\nt={state_times[num_step]:.3f}"
            ax_pred.set_title(pred_title, fontsize=10)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            ax_pred.set_aspect("equal")

        for col in range(time_steps, steps_per_figure):
            axes[0, col].axis("off")
            axes[1, col].axis("off")

        if contour_ref is not None:
            cbar = fig.colorbar(
                contour_ref,
                ax=axes.ravel().tolist(),
                fraction=0.06,
                pad=0.02,
            )
            cbar.set_label("Temperature (T)")

        output_path = output_dir / f"{output_prefix}_pred_true_compare_part{fig_idx}_{ind}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Prediction visualization saved to {output_path}")


def main() -> None:
    args, overrides = parse_args()
    set_random_seed(args.seed)
    load_dir = args.load_dir.resolve()
    state_dict, config, checkpoint_path = load_checkpoint_and_config(
        load_dir, args.checkpoint, overrides=overrides
    )
    if args.out_time_window is not None:
        if args.out_time_window <= 0:
            raise ValueError(f"out_time_window must be > 0, got {args.out_time_window}.")
        original_out_time_window = getattr(config.model, "out_time_window", None)
        config.model.out_time_window = args.out_time_window
        print(
            f"Overriding config.model.out_time_window: "
            f"{original_out_time_window} -> {config.model.out_time_window}"
        )
    if args.viz_interval <= 0:
        raise ValueError(f"viz_interval must be > 0, got {args.viz_interval}.")
    model_type = infer_model_type(config, args.model_type)

    target_step = args.target_step if args.target_step is not None else config.model.out_time_window - 1
    if not 0 <= target_step < config.model.out_time_window:
        raise ValueError(
            f"target_step must be within [0, {config.model.out_time_window - 1}], got {target_step}."
        )
    results_root = Path(__file__).resolve().parent / "result_RB"
    analysis_name = build_analysis_name(
        model_type=model_type,
        load_dir=load_dir,
        checkpoint_path=checkpoint_path,
        sample_index=args.sample_index,
        target_step=target_step,
        physical_io=args.physical_io,
    )
    output_dir = create_unique_output_dir(results_root, analysis_name)
    output_prefix = sanitize_path_component(
        f"{model_type}_{infer_output_prefix(checkpoint_path)}_{'physical' if args.physical_io else 'normalized'}"
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 60)
    print("Starting Sensitivity Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    if overrides:
        print(f"Config overrides: {overrides}")
    print(
        "Sensitivity mode: "
        + ("physical input/output graph" if args.physical_io else "normalized input/output graph")
    )

    data_files = select_data_files(config)
    normalization_files = None
    dataset = RB2D_Dataset(
        data_folder=config.data.data_folder,
        data_filenames=data_files,
        nx=config.data.nx,
        ny=config.data.ny,
        nt_in=config.model.in_time_window,
        nt_out=config.model.out_time_window,
        stride_t=config.data.stride,
        normalize_channels=False,
        subsample_rate=config.data.subsample_rate,
        BC=False,
    )
    if config.data.normalize:
        normalization_files = select_normalization_files(config)
        normalization_dataset = RB2D_Dataset(
            data_folder=config.data.data_folder,
            data_filenames=normalization_files,
            nx=config.data.nx,
            ny=config.data.ny,
            nt_in=config.model.in_time_window,
            nt_out=config.model.out_time_window,
            stride_t=config.data.stride,
            normalize_channels=True,
            subsample_rate=config.data.subsample_rate,
            BC=False,
        )
        dataset.normalize_channels = True
        dataset._mean = normalization_dataset.channel_mean.copy()
        dataset._std = normalization_dataset.channel_std.copy()
        print(
            "Using normalization stats from train-preferred files: "
            f"{normalization_files}"
        )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index {args.sample_index} out of range (len={len(dataset)}).")

    input_clip, label_clip, _t_coord, x_coord, y_coord = dataset[args.sample_index]
    input_clip = input_clip.unsqueeze(0).to(device)  # [1, t_in, nx, ny, c]
    label_clip = label_clip.unsqueeze(0).to(device)  # [1, t_out, nx, ny, c]
    model_x_coords = np.asarray(x_coord.detach().cpu().numpy(), dtype=np.float64)
    model_y_coords = np.asarray(y_coord.detach().cpu().numpy(), dtype=np.float64)

    nx, ny = config.data.nx, config.data.ny
    roi_x_start, roi_x_end, roi_y_start, roi_y_end = compute_roi_bounds(
        nx, ny, args.roi_size, args.roi_x_start, args.roi_y_start
    )
    model_roi_bounds_physical = roi_bounds_from_coords(
        model_x_coords,
        model_y_coords,
        roi_x_start,
        roi_x_end,
        roi_y_start,
        roi_y_end,
    )

    pos_lst = build_pos_lst_from_coords(x_coord, y_coord, device)

    model = build_model(config, model_type).to(device)
    load_state_dict_flexible(model, state_dict)
    model.eval()
    adjoint_grad_data, adjoint_grad_times, adjoint_step_labels, adjoint_roi_bounds, adjoint_path = load_adjoint_gradient_data(
        args.adjoint_path
    )
    adjoint_temp_absmax = float(np.abs(adjoint_grad_data[..., 0]).max())
    if adjoint_temp_absmax <= 0:
        adjoint_temp_absmax = 1e-8
    adjoint_x_coords, adjoint_z_coords = build_adjoint_coords(
        adjoint_grad_data.shape[1],
        adjoint_grad_data.shape[2],
    )

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Model type: {model_type}")
    print(f"Model params: {count_params(model)}")
    print(f"Loaded adjoint comparison from {adjoint_path}")
    print(f"Adjoint temperature gradient absmax: {adjoint_temp_absmax:.6e}")

    predict_steps = config.model.out_time_window
    input_times, pred_times, state_times = build_rollout_times(
        dataset=dataset,
        sample_index=args.sample_index,
        nt_in=config.model.in_time_window,
        predict_steps=predict_steps,
    )
    state_step_labels = build_rollout_labels(
        nt_in=config.model.in_time_window,
        predict_steps=predict_steps,
    )
    input_clip_phys = dataset.denormalize_grid(input_clip)
    label_clip_phys = dataset.denormalize_grid(label_clip)
    initial_input_phys = None
    if args.physical_io:
        initial_input_phys = input_clip_phys.clone().detach().requires_grad_(True)
        initial_input = dataset.normalize_grid(initial_input_phys)
    else:
        initial_input = input_clip.clone().detach().requires_grad_(True)
    x_cur = initial_input
    pred_sequence = []

    for step in range(predict_steps):
        y_pred_frame = rollout_step(
            model=model,
            x_cur=x_cur,
            pos_lst=pos_lst,
            model_type=model_type,
            in_time_window=config.model.in_time_window,
        )
        y_pred_step_norm = y_pred_frame.unsqueeze(1)
        if args.physical_io:
            y_pred_step = dataset.denormalize_grid(y_pred_step_norm)
            y_pred_step_for_rollout = dataset.normalize_grid(y_pred_step)
        else:
            y_pred_step = y_pred_step_norm
            y_pred_step_for_rollout = y_pred_step
        y_pred_step.retain_grad()
        pred_sequence.append(y_pred_step)
        x_cur = update_autoregressive_input(
            x_cur=x_cur,
            y_pred_step=y_pred_step_for_rollout,
            in_time_window=config.model.in_time_window,
        )
        print(f"Generated step {step + 1}/{predict_steps}")

    target_frame = pred_sequence[target_step]
    # print('target_frame shape:', target_frame.shape)
    target_region = target_frame[:, 0, roi_x_start:roi_x_end, roi_y_start:roi_y_end, args.roi_channel]
    target_value = target_region.mean()
    print(
        f"Target: step {target_step + 1}, channel {args.roi_channel}, "
        f"x[{roi_x_start}:{roi_x_end}], y[{roi_y_start}:{roi_y_end}] -> {target_value.item():.6f}"
    )

    model.zero_grad(set_to_none=True)
    grad_source = initial_input_phys if args.physical_io else initial_input
    if grad_source.grad is not None:
        grad_source.grad.zero_()
    target_value.backward()

    input_grad = grad_source.grad
    if input_grad is None:
        print("No gradient captured for the initial input.")
    else:
        for step_idx in range(input_grad.shape[1]):
            grad_val = input_grad[:, step_idx]
            if args.physical_io:
                print(f"  Input Step {step_idx} g_phys mean: {grad_val.mean().item():.6e}")
            else:
                print(f"  Input Step {step_idx}: {grad_val.mean().item():.6e}")

    print(
        "Predicted step physical gradient norms:"
        if args.physical_io
        else "Predicted step gradient norms:"
    )
    for step_idx, frame in enumerate(pred_sequence):
        grad = frame.grad
        label = f"Pred Step {step_idx + 1}"
        if grad is None:
            print(f"  {label}: gradient is None")
            continue
        suffix = " (target)" if step_idx == target_step else ""
        if args.physical_io:
            print(f"  {label}{suffix} g_phys mean: {grad.mean().item():.6e}")
        else:
            print(f"  {label}{suffix}: {grad.mean().item():.6e}")

    grad_frames = []
    step_labels = []
    grad_times = []

    if input_grad is not None:
        for t_idx in range(input_grad.shape[1]):
            grad_frames.append(input_grad[:, t_idx, ...])
            step_labels.append(f"Input Step {t_idx + 1}")
            if t_idx < len(input_times):
                grad_times.append(float(input_times[t_idx]))

    for step_idx, frame in enumerate(pred_sequence):
        grad = frame.grad
        if grad is None:
            continue
        grad_frames.append(grad[:, 0, ...])
        label = f"Pred Step {step_idx + 1}"
        if step_idx == target_step:
            label += " (target)"
        step_labels.append(label)
        if step_idx < len(pred_times):
            grad_times.append(float(pred_times[step_idx]))

    if grad_frames:
        gradient_stack = torch.stack(grad_frames, dim=1)
        gradients_np = gradient_stack[0].detach().cpu().numpy()

        plot_grad_pred(
            gradients_np,
            nx,
            ny,
            step_labels,
            np.asarray(grad_times, dtype=np.float64),
            load_dir,
            output_prefix,
            model_roi_bounds_physical,
            type='grad',
            ind=args.sample_index,
            viz_interval=args.viz_interval,
            temp_absmax_override=adjoint_temp_absmax,
            x_coords=model_x_coords,
            y_coords=model_y_coords,
            transpose_fields=True,
            roi_bounds_are_indices=False,
        )
        print(
            "Physical-gradient visualization completed."
            if args.physical_io
            else "Gradient visualization completed."
        )
    else:
        print("No gradients captured for visualization.")
        gradients_np = np.empty((0, nx, ny, input_clip.shape[-1]), dtype=np.float32)

    plot_grad_pred(
        adjoint_grad_data,
        adjoint_grad_data.shape[1],
        adjoint_grad_data.shape[2],
        adjoint_step_labels,
        adjoint_grad_times,
        load_dir,
        output_prefix,
        adjoint_roi_bounds if adjoint_roi_bounds is not None else model_roi_bounds_physical,
        type='adjoint',
        ind=args.sample_index,
        viz_interval=10,
        temp_absmax_override=adjoint_temp_absmax,
        invert_vertical=True,
        x_coords=adjoint_x_coords,
        y_coords=adjoint_z_coords,
        transpose_fields=False,
        roi_bounds_are_indices=False,
    )
    print("Adjoint-gradient visualization completed.")

    with torch.no_grad():
        predict_future = torch.cat([frame.detach() for frame in pred_sequence], dim=1)
        if args.physical_io:
            assert initial_input_phys is not None
            predict_full = torch.cat([initial_input_phys.detach(), predict_future], dim=1)
            label_full = torch.cat([initial_input_phys.detach(), label_clip_phys], dim=1)
        else:
            predict_full = torch.cat([initial_input.detach(), predict_future], dim=1)
            label_full = torch.cat([initial_input.detach(), label_clip], dim=1)
        if config.data.normalize and not args.physical_io:
            predict_full = dataset.denormalize_grid(predict_full)
            label_full = dataset.denormalize_grid(label_full)
    
    print(f"Prediction visualization")

    predictions_np = predict_full[0].detach().cpu().numpy()
    label_full_np = label_full[0].detach().cpu().numpy()

    plot_grad_pred(
        predictions_np,
        nx,
        ny,
        state_step_labels,
        state_times,
        load_dir,
        output_prefix,
        (roi_x_start, roi_x_end, roi_y_start, roi_y_end),
        type='pred',
        ind=args.sample_index,
        viz_interval=args.viz_interval,
    )
    print(f"Prediction visualization completed.")

    plot_grad_pred(
        label_full_np,
        nx,
        ny,
        state_step_labels,
        state_times,
        load_dir,
        output_prefix,
        (roi_x_start, roi_x_end, roi_y_start, roi_y_end),
        type='true',
        ind=args.sample_index,
        viz_interval=args.viz_interval,
    )
    print("Ground-truth visualization completed.")

    plot_pred_true_compare(
        predictions_np,
        label_full_np,
        nx,
        ny,
        state_times,
        load_dir,
        output_prefix,
        ind=args.sample_index,
        steps_per_figure=10,
    )
    input_count = int(config.model.in_time_window)
    sensitivity_space = "physical" if args.physical_io else "normalized"
    save_analysis_artifacts(
        output_dir=output_dir,
        output_prefix=output_prefix,
        input_states_physical=input_clip_phys[0].detach().cpu().numpy(),
        ground_truth_future_physical=label_full_np[input_count:],
        prediction_future_physical=predictions_np[input_count:],
        ground_truth_full_physical=label_full_np,
        prediction_full_physical=predictions_np,
        sensitivity=gradients_np,
        sensitivity_space=sensitivity_space,
        input_times=input_times,
        pred_times=pred_times,
        state_times=state_times,
        sensitivity_times=np.asarray(grad_times, dtype=np.float64),
        model_x_coords=model_x_coords,
        model_y_coords=model_y_coords,
        roi_bounds_index=(roi_x_start, roi_x_end, roi_y_start, roi_y_end),
        roi_bounds_physical=model_roi_bounds_physical,
        state_step_labels=state_step_labels,
        sensitivity_step_labels=step_labels,
        data_files=data_files,
        normalization_files=normalization_files,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        load_dir=load_dir,
        sample_index=args.sample_index,
        target_step=target_step,
        roi_channel=args.roi_channel,
        physical_io=args.physical_io,
    )
    print("Analysis arrays and metadata saved.")

    
        


if __name__ == "__main__":
    main()

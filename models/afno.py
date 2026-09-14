from __future__ import annotations

import math
from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _to_2tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple patch/image size, got {value}.")
        return int(value[0]), int(value[1])
    scalar = int(value)
    return scalar, scalar


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, std=std)
    return nn.init.normal_(tensor, std=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        in_chans: int,
        embed_dim: int,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)

        if self.img_size[0] % self.patch_size[0] != 0 or self.img_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"img_size={self.img_size} must be divisible by patch_size={self.patch_size}."
            )

        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size ({height}, {width}) does not match model {self.img_size}."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    """
    AFNO spectral mixer compatible with:
    - [B, N, C] tokens with optional spatial_size
    - [B, H, W, C] spatial grids
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}."
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2, self.num_blocks, self.block_size, self.block_size * hidden_size_factor
            )
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2, self.num_blocks, self.block_size * hidden_size_factor, self.block_size
            )
        )
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def _resolve_layout(
        self, x: torch.Tensor, spatial_size: Optional[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, Tuple[int, int], bool]:
        if x.ndim == 4:
            bsz, height, width, channels = x.shape
            if channels != self.hidden_size:
                raise ValueError(
                    f"Expected channel dim {self.hidden_size}, got {channels}."
                )
            return x, (height, width), True

        if x.ndim == 3:
            bsz, n_tokens, channels = x.shape
            if channels != self.hidden_size:
                raise ValueError(
                    f"Expected channel dim {self.hidden_size}, got {channels}."
                )
            if spatial_size is None:
                side = int(math.sqrt(n_tokens))
                if side * side != n_tokens:
                    raise ValueError(
                        f"Cannot infer square spatial_size from token count {n_tokens}. "
                        "Pass spatial_size=(H, W)."
                    )
                spatial_size = (side, side)
            if spatial_size[0] * spatial_size[1] != n_tokens:
                raise ValueError(
                    f"spatial_size={spatial_size} does not match token count {n_tokens}."
                )
            return x.reshape(bsz, spatial_size[0], spatial_size[1], channels), spatial_size, False

        raise ValueError(f"Expected input rank 3 or 4, got shape {tuple(x.shape)}.")

    def forward(
        self, x: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        x_grid, (height, width), input_was_grid = self._resolve_layout(x, spatial_size)
        bias = x_grid

        dtype = x_grid.dtype
        x_freq = torch.fft.rfft2(x_grid.float(), dim=(1, 2), norm="ortho")
        freq_w = x_freq.shape[2]
        x_freq = x_freq.reshape(
            x_grid.shape[0], height, freq_w, self.num_blocks, self.block_size
        )

        o1_real = x_freq.real.new_zeros(
            x_freq.shape[0],
            height,
            freq_w,
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = x_freq.real.new_zeros(x_freq.shape)
        o2_imag = x_freq.real.new_zeros(x_freq.shape)

        total_modes = height // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        kept_modes = max(1, min(total_modes, kept_modes))
        row_start = max(0, total_modes - kept_modes)
        row_end = min(height, total_modes + kept_modes)
        kept_w = min(freq_w, kept_modes)

        if row_start < row_end and kept_w > 0:
            x_real = x_freq[:, row_start:row_end, :kept_w].real
            x_imag = x_freq[:, row_start:row_end, :kept_w].imag

            o1_real[:, row_start:row_end, :kept_w] = F.relu(
                torch.einsum("...bi,bio->...bo", x_real, self.w1[0])
                - torch.einsum("...bi,bio->...bo", x_imag, self.w1[1])
                + self.b1[0]
            )
            o1_imag[:, row_start:row_end, :kept_w] = F.relu(
                torch.einsum("...bi,bio->...bo", x_imag, self.w1[0])
                + torch.einsum("...bi,bio->...bo", x_real, self.w1[1])
                + self.b1[1]
            )

            o2_real[:, row_start:row_end, :kept_w] = (
                torch.einsum("...bi,bio->...bo", o1_real[:, row_start:row_end, :kept_w], self.w2[0])
                - torch.einsum("...bi,bio->...bo", o1_imag[:, row_start:row_end, :kept_w], self.w2[1])
                + self.b2[0]
            )
            o2_imag[:, row_start:row_end, :kept_w] = (
                torch.einsum("...bi,bio->...bo", o1_imag[:, row_start:row_end, :kept_w], self.w2[0])
                + torch.einsum("...bi,bio->...bo", o1_real[:, row_start:row_end, :kept_w], self.w2[1])
                + self.b2[1]
            )

        x_out = torch.stack((o2_real, o2_imag), dim=-1)
        x_out = F.softshrink(x_out, lambd=self.sparsity_threshold)
        x_out = torch.view_as_complex(x_out)
        x_out = x_out.reshape(x_grid.shape[0], height, freq_w, self.hidden_size)
        x_out = torch.fft.irfft2(x_out, s=(height, width), dim=(1, 2), norm="ortho")
        x_out = x_out.to(dtype) + bias

        if input_was_grid:
            return x_out
        return x_out.reshape(x_grid.shape[0], height * width, self.hidden_size)


class AFNO2DMixer(AFNO2D):
    pass


class AFNOBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
        double_skip: bool = True,
        norm_layer: Any = None,
    ):
        super().__init__()
        norm_layer = norm_layer or nn.LayerNorm
        self.norm1 = norm_layer(hidden_size)
        self.filter = AFNO2D(
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            hidden_size_factor=hidden_size_factor,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(hidden_size)
        self.mlp = MLP(dim=hidden_size, mlp_ratio=mlp_ratio, drop=drop)
        self.double_skip = double_skip

    def forward(
        self, x: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.filter(x, spatial_size=spatial_size)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNOModel(nn.Module):
    """
    FourCastNet-style AFNO backbone wrapped for this repo:
    - input:  [B, T, X, Y, C] or [B, X, Y, C]
    - output: [B, X, Y, out_channels]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_time_window: int = 1,
        img_size: Tuple[int, int] = (128, 64),
        patch_size: Any = 1,
        hidden_size: int = 64,
        depth: int = 4,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        uniform_drop: bool = False,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
        double_skip: bool = True,
        use_pos_embed: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_time_window = int(in_time_window)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.use_pos_embed = bool(use_pos_embed)
        self.activation_checkpointing = bool(activation_checkpointing)

        flat_in_channels = self.in_time_window * self.in_channels
        self.num_features = self.embed_dim = hidden_size
        self.num_blocks = num_blocks

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=flat_in_channels,
            embed_dim=hidden_size,
        )

        self.h, self.w = self.patch_embed.grid_size
        num_patches = self.patch_embed.num_patches

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
            _trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    hidden_size=hidden_size,
                    num_blocks=num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[idx],
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    hidden_size_factor=hidden_size_factor,
                    double_skip=double_skip,
                    norm_layer=norm_layer,
                )
                for idx in range(depth)
            ]
        )
        self.norm = norm_layer(hidden_size)
        patch_area = self.patch_size[0] * self.patch_size[1]
        self.head = nn.Linear(hidden_size, self.out_channels * patch_area, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.activation_checkpointing or not self.training:
            return block(x)
        try:
            return checkpoint(block, x, use_reentrant=False)
        except TypeError:
            return checkpoint(block, x)

    def _prepare_input(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 5:
            bsz, steps, nx, ny, channels = u.shape
            if steps != self.in_time_window:
                raise ValueError(
                    f"Expected in_time_window={self.in_time_window}, got {steps}."
                )
            if channels != self.in_channels:
                raise ValueError(
                    f"Expected in_channels={self.in_channels}, got {channels}."
                )
            x = u.permute(0, 1, 4, 2, 3).reshape(bsz, steps * channels, nx, ny)
        elif u.ndim == 4:
            bsz, nx, ny, channels = u.shape
            if self.in_time_window != 1:
                raise ValueError("4D input is only valid when in_time_window=1.")
            if channels != self.in_channels:
                raise ValueError(
                    f"Expected in_channels={self.in_channels}, got {channels}."
                )
            x = u.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {tuple(u.shape)}")

        if (nx, ny) != self.img_size:
            raise ValueError(
                f"Input spatial size ({nx}, {ny}) does not match model img_size={self.img_size}."
            )
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(bsz, self.h, self.w, self.embed_dim)

        for block in self.blocks:
            x = self._run_block(block, x)

        x = self.norm(x)
        return x

    def forward(self, u: torch.Tensor, pos_lst: Optional[Any] = None) -> torch.Tensor:
        del pos_lst  # API compatibility with IFactFormer/FNO training code.

        x = self._prepare_input(u)
        x = self.forward_features(x)
        x = self.head(x)

        bsz, h_grid, w_grid, _ = x.shape
        p1, p2 = self.patch_size
        x = x.reshape(bsz, h_grid, w_grid, p1, p2, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            bsz,
            h_grid * p1,
            w_grid * p2,
            self.out_channels,
        )
        return x


def build_afno_from_config(config: Any) -> AFNOModel:
    model_cfg = config.model
    in_channels = getattr(model_cfg, "in_channels", getattr(model_cfg, "in_dim", None))
    out_channels = getattr(model_cfg, "out_channels", getattr(model_cfg, "out_dim", None))
    if in_channels is None or out_channels is None:
        raise ValueError(
            "Cannot infer AFNO in/out channels from config.model. "
            "Expected one of (in_channels/out_channels) or (in_dim/out_dim)."
        )

    afno_kwargs = {}
    if hasattr(model_cfg, "afno_kwargs") and model_cfg.afno_kwargs is not None:
        if isinstance(model_cfg.afno_kwargs, dict):
            afno_kwargs = dict(model_cfg.afno_kwargs)
        else:
            afno_kwargs = vars(model_cfg.afno_kwargs)

    img_size = getattr(model_cfg, "img_size", None)
    if img_size is None:
        if not hasattr(config, "data") or not hasattr(config.data, "nx") or not hasattr(config.data, "ny"):
            raise ValueError(
                "AFNOModel requires img_size or config.data.{nx, ny} to build PatchEmbed."
            )
        img_size = (config.data.nx, config.data.ny)

    init_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "in_time_window": getattr(model_cfg, "in_time_window", 1),
        "img_size": img_size,
        "patch_size": getattr(model_cfg, "patch_size", 1),
        "hidden_size": getattr(model_cfg, "hidden_size", getattr(model_cfg, "dim", 64)),
        "depth": getattr(model_cfg, "depth", 4),
        "num_blocks": getattr(model_cfg, "num_blocks", 8),
        "mlp_ratio": getattr(model_cfg, "mlp_ratio", 4.0),
        "drop_rate": getattr(model_cfg, "drop_rate", 0.0),
        "drop_path_rate": getattr(model_cfg, "drop_path_rate", 0.0),
        "uniform_drop": getattr(model_cfg, "uniform_drop", False),
        "sparsity_threshold": getattr(model_cfg, "sparsity_threshold", 0.01),
        "hard_thresholding_fraction": getattr(model_cfg, "hard_thresholding_fraction", 1.0),
        "hidden_size_factor": getattr(model_cfg, "hidden_size_factor", 1),
        "double_skip": getattr(model_cfg, "double_skip", True),
        "use_pos_embed": getattr(model_cfg, "use_pos_embed", True),
        "activation_checkpointing": getattr(model_cfg, "activation_checkpointing", False),
    }
    init_kwargs.update(afno_kwargs)
    return AFNOModel(**init_kwargs)

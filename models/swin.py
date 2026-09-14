from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _meshgrid_ij(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.meshgrid(x, y, indexing="ij")
    except TypeError:
        return torch.meshgrid(x, y)


def _to_2tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple/list, got {value}.")
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


def _to_stage_list(value: Any, num_stages: int, name: str) -> List[int]:
    if isinstance(value, (tuple, list)):
        values = [int(v) for v in value]
        if len(values) != num_stages:
            raise ValueError(
                f"{name} length must match number of stages {num_stages}, got {values}."
            )
        return values
    return [int(value) for _ in range(num_stages)]


def _format_stage_spec(spec: Sequence[int]) -> str:
    return "-".join(str(v) for v in spec)


def _to_stage_window_list(
    value: Any,
    num_stages: int,
    name: str = "window_sizes",
) -> List[Tuple[int, int]]:
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(not isinstance(v, (tuple, list)) for v in value):
            return [_to_2tuple(value) for _ in range(num_stages)]

        values = list(value)
        if len(values) != num_stages:
            raise ValueError(
                f"{name} length must match number of stages {num_stages}, got {values}."
            )
        return [_to_2tuple(v) for v in values]

    return [_to_2tuple(value) for _ in range(num_stages)]


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


def window_partition(
    x: torch.Tensor, window_size: int | Tuple[int, int]
) -> torch.Tensor:
    window_size = _to_2tuple(window_size)
    window_height, window_width = window_size
    batch_size, height, width, channels = x.shape
    x = x.reshape(
        batch_size,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.reshape(-1, window_height * window_width, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: int | Tuple[int, int],
    height: int,
    width: int,
) -> torch.Tensor:
    window_size = _to_2tuple(window_size)
    window_height, window_width = window_size
    batch_size = windows.shape[0] // ((height // window_height) * (width // window_width))
    x = windows.reshape(
        batch_size,
        height // window_height,
        width // window_width,
        window_height,
        window_width,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(batch_size, height, width, -1)


class WindowAttentionV2(nn.Module):
    """
    Swin Transformer V2 attention core with continuous relative position bias.
    The relative bias is computed on the fly so the module can handle arbitrary
    PDE grid sizes after runtime padding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: int | Tuple[int, int] = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pretrained_window_size = _to_2tuple(pretrained_window_size)

        self.logit_scale = nn.Parameter(torch.log(10.0 * torch.ones(num_heads, 1, 1)))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _relative_position_bias(
        self, window_shape: Tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        window_height, window_width = window_shape

        relative_coords_h = torch.arange(
            -(window_height - 1), window_height, dtype=torch.float32, device=device
        )
        relative_coords_w = torch.arange(
            -(window_width - 1), window_width, dtype=torch.float32, device=device
        )
        table_h, table_w = _meshgrid_ij(relative_coords_h, relative_coords_w)
        relative_coords_table = torch.stack((table_h, table_w), dim=-1).unsqueeze(0)

        ref_height = (
            self.pretrained_window_size[0]
            if self.pretrained_window_size[0] > 0
            else window_height
        )
        ref_width = (
            self.pretrained_window_size[1]
            if self.pretrained_window_size[1] > 0
            else window_width
        )
        relative_coords_table[..., 0] /= max(ref_height - 1, 1)
        relative_coords_table[..., 1] /= max(ref_width - 1, 1)
        relative_coords_table *= 8.0
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / math.log2(8.0)

        coords_h = torch.arange(window_height, device=device)
        coords_w = torch.arange(window_width, device=device)
        coords = torch.stack(_meshgrid_ij(coords_h, coords_w))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[..., 0] += window_height - 1
        relative_coords[..., 1] += window_width - 1
        relative_coords[..., 0] *= 2 * window_width - 1
        relative_position_index = relative_coords.sum(-1)

        relative_position_bias_table = self.cpb_mlp(relative_coords_table).reshape(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.reshape(
            window_height * window_width,
            window_height * window_width,
            self.num_heads,
        ).permute(2, 0, 1)
        relative_position_bias = 16.0 * torch.sigmoid(relative_position_bias)
        return relative_position_bias.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        window_shape: Tuple[int, int],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_windows, n_tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected token dim {self.dim}, got {channels} in window attention."
            )

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale.to(dtype=attn.dtype)
        attn = attn + self._relative_position_bias(
            window_shape=window_shape,
            device=x.device,
            dtype=attn.dtype,
        ).unsqueeze(0)

        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.reshape(
                batch_windows // n_windows, n_windows, self.num_heads, n_tokens, n_tokens
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, n_tokens, n_tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, n_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalFrequencyMixer(nn.Module):
    """
    Optional low-frequency residual branch added on top of Swin V2 blocks.
    It preserves the local shifted-window attention while reintroducing a
    lightweight global path for long-range operator interactions.
    """

    def __init__(
        self,
        dim: int,
        modes_height: int = 16,
        modes_width: int = 16,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.modes_height = max(1, int(modes_height))
        self.modes_width = max(1, int(modes_width))
        scale = 0.02
        self.weight_top = nn.Parameter(
            scale * torch.randn(dim, self.modes_height, self.modes_width, 2)
        )
        self.weight_bottom = nn.Parameter(
            scale * torch.randn(dim, self.modes_height, self.modes_width, 2)
        )
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected channel dim {self.dim}, got {channels} in spectral mixer."
            )

        dtype = x.dtype
        x_freq = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        out_freq = torch.zeros_like(x_freq)

        kept_height = min(self.modes_height, height)
        kept_width = min(self.modes_width, x_freq.shape[2])

        weight_top = torch.complex(
            self.weight_top[:, :kept_height, :kept_width, 0],
            self.weight_top[:, :kept_height, :kept_width, 1],
        ).permute(1, 2, 0)
        out_freq[:, :kept_height, :kept_width, :] = (
            x_freq[:, :kept_height, :kept_width, :] * weight_top
        )

        bottom_height = max(kept_height - 1, 0)
        if bottom_height > 0:
            weight_bottom = torch.complex(
                self.weight_bottom[:, :bottom_height, :kept_width, 0],
                self.weight_bottom[:, :bottom_height, :kept_width, 1],
            ).permute(1, 2, 0)
            out_freq[:, -bottom_height:, :kept_width, :] = (
                x_freq[:, -bottom_height:, :kept_width, :] * weight_bottom
            )

        x = torch.fft.irfft2(out_freq, s=(height, width), dim=(1, 2), norm="ortho")
        x = self.output_proj(x.to(dtype))
        x = self.dropout(x)
        return x


class SwinOperatorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int | Tuple[int, int] = 8,
        shift_size: int | Tuple[int, int] = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_global_mixer: bool = False,
        global_modes: Tuple[int, int] = (16, 16),
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.shift_size = _to_2tuple(shift_size)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=self.window_size,
        )

        self.use_global_mixer = bool(use_global_mixer)
        if self.use_global_mixer:
            self.norm_global = nn.LayerNorm(dim)
            self.global_mixer = GlobalFrequencyMixer(
                dim=dim,
                modes_height=global_modes[0],
                modes_width=global_modes[1],
                drop=drop,
            )
        else:
            self.norm_global = None
            self.global_mixer = None

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if layer_scale_init is not None and layer_scale_init > 0.0:
            self.gamma_attn = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma_mlp = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma_global = (
                nn.Parameter(layer_scale_init * torch.ones(dim))
                if self.use_global_mixer
                else None
            )
        else:
            self.gamma_attn = None
            self.gamma_mlp = None
            self.gamma_global = None

    @staticmethod
    def _apply_layer_scale(
        x: torch.Tensor, gamma: Optional[torch.nn.Parameter]
    ) -> torch.Tensor:
        if gamma is None:
            return x
        return x * gamma.view(1, 1, 1, -1)

    @staticmethod
    def _build_attention_mask(
        height: int,
        width: int,
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        window_height, window_width = window_size
        shift_height, shift_width = shift_size
        img_mask = torch.zeros((1, height, width, 1), device=device, dtype=torch.float32)
        h_slices = (
            slice(0, -window_height),
            slice(-window_height, -shift_height),
            slice(-shift_height, None),
        )
        w_slices = (
            slice(0, -window_width),
            slice(-window_width, -shift_width),
            slice(-shift_width, None),
        )

        counter = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = counter
                counter += 1

        mask_windows = window_partition(img_mask, window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask.to(dtype=dtype)

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected channel dim {self.dim}, got {channels} in Swin block."
            )

        window_size = (
            max(1, min(self.window_size[0], height)),
            max(1, min(self.window_size[1], width)),
        )
        if window_size[0] < 1 or window_size[1] < 1:
            raise ValueError(f"Invalid window size {window_size} for shape {tuple(x.shape)}.")
        shift_size = (
            0 if window_size[0] <= 1 else min(self.shift_size[0], window_size[0] // 2),
            0 if window_size[1] <= 1 else min(self.shift_size[1], window_size[1] // 2),
        )

        pad_height = (window_size[0] - height % window_size[0]) % window_size[0]
        pad_width = (window_size[1] - width % window_size[1]) % window_size[1]
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))

        padded_height, padded_width = x.shape[1], x.shape[2]
        attn_mask = None

        if shift_size[0] > 0 or shift_size[1] > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = self._build_attention_mask(
                height=padded_height,
                width=padded_width,
                window_size=window_size,
                shift_size=shift_size,
                device=x.device,
                dtype=x.dtype,
            )

        x_windows = window_partition(x, window_size)
        attn_windows = self.attn(
            x_windows,
            window_shape=window_size,
            mask=attn_mask,
        )
        x = window_reverse(attn_windows, window_size, padded_height, padded_width)

        if shift_size[0] > 0 or shift_size[1] > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        if pad_height > 0 or pad_width > 0:
            x = x[:, :height, :width, :].contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self._window_attention(self.norm1(x))
        x = x + self.drop_path(self._apply_layer_scale(attn_out, self.gamma_attn))

        if self.use_global_mixer and self.global_mixer is not None and self.norm_global is not None:
            global_out = self.global_mixer(self.norm_global(x))
            x = x + self.drop_path(
                self._apply_layer_scale(global_out, self.gamma_global)
            )

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(self._apply_layer_scale(mlp_out, self.gamma_mlp))
        return x


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int] = 2,
        norm: bool = True,
    ):
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim) if norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected channel dim {self.dim}, got {channels} in patch merging."
            )

        pad_height = height % 2
        pad_width = width % 2
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        scale_factor: int | Tuple[int, int] = 2,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.scale_factor = _to_2tuple(scale_factor)
        scale_height, scale_width = self.scale_factor
        self.expand = nn.Linear(dim, scale_height * scale_width * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected channel dim {self.dim}, got {channels} in patch expand."
            )

        scale_height, scale_width = self.scale_factor
        x = self.expand(x)
        x = x.reshape(
            batch_size,
            height,
            width,
            scale_height,
            scale_width,
            self.out_dim,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(
            batch_size,
            height * scale_height,
            width * scale_width,
            self.out_dim,
        )
        x = self.norm(x)
        return x


class EncoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rates: Sequence[float],
        global_flags: Sequence[bool],
        global_modes: Tuple[int, int],
        layer_scale_init: float,
        downsample: bool,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(drop_path_rates) != depth or len(global_flags) != depth:
            raise ValueError("drop_path_rates/global_flags length must equal stage depth.")

        self.use_checkpoint = bool(use_checkpoint)
        self.blocks = nn.ModuleList(
            [
                SwinOperatorBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        (0, 0)
                        if block_idx % 2 == 0
                        else (window_size[0] // 2, window_size[1] // 2)
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[block_idx],
                    use_global_mixer=global_flags[block_idx],
                    global_modes=global_modes,
                    layer_scale_init=layer_scale_init,
                )
                for block_idx in range(depth)
            ]
        )
        self.downsample = PatchMerging2D(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return skip, x


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rates: Sequence[float],
        global_flags: Sequence[bool],
        global_modes: Tuple[int, int],
        layer_scale_init: float,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(drop_path_rates) != depth or len(global_flags) != depth:
            raise ValueError("drop_path_rates/global_flags length must equal stage depth.")

        self.use_checkpoint = bool(use_checkpoint)
        self.upsample = PatchExpand2D(in_dim, out_dim=skip_dim, scale_factor=2)
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * skip_dim),
            nn.Linear(2 * skip_dim, skip_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                SwinOperatorBlock(
                    dim=skip_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        (0, 0)
                        if block_idx % 2 == 0
                        else (window_size[0] // 2, window_size[1] // 2)
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[block_idx],
                    use_global_mixer=global_flags[block_idx],
                    global_modes=global_modes,
                    layer_scale_init=layer_scale_init,
                )
                for block_idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = x[:, : skip.shape[1], : skip.shape[2], :]
            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                raise ValueError(
                    f"Skip shape {tuple(skip.shape)} does not match upsampled shape {tuple(x.shape)}."
                )
        x = torch.cat([x, skip], dim=-1)
        x = self.fuse(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class SwinOperatorModel(nn.Module):
    """
    Hierarchical Swin V2-style encoder-decoder neural operator.

    Relative to the official classification architecture, this version keeps:
    - patch embedding
    - hierarchical Swin stages with shifted-window attention
    - patch merging downsampling

    And adapts it for PDE/operator learning by adding:
    - mirrored patch-expansion decoder for dense full-resolution prediction
    - optional physical-coordinate patch embedding
    - residual delta prediction for stable autoregressive rollout
    - optional low-frequency global mixer inside selected blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_time_window: int = 1,
        hidden_size: int = 64,
        patch_size: int | Tuple[int, int] = 2,
        depths: Sequence[int] = (2, 2, 4, 2),
        num_heads: Sequence[int] = (4, 8, 16, 32),
        window_sizes: Sequence[int] | int | Tuple[int, int] = 8,
        decoder_depths: Optional[Sequence[int]] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        patch_norm: bool = True,
        full_pos_embed: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        coord_embed: bool = True,
        predict_delta: bool = True,
        use_global_mixer: bool = False,
        global_mixer_every: int = 1,
        global_modes: Tuple[int, int] = (16, 16),
        layer_scale_init: float = 1e-3,
        use_checkpoint: bool = False,
        input_grid_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.in_time_window = int(in_time_window)
        self.hidden_size = int(hidden_size)
        self.patch_size = _to_2tuple(patch_size)
        self.depths = [int(v) for v in depths]
        if len(self.depths) == 0:
            raise ValueError("depths must contain at least one stage.")
        self.num_stages = len(self.depths)
        self.num_heads = _to_stage_list(num_heads, self.num_stages, "num_heads")
        self.window_sizes = _to_stage_window_list(window_sizes, self.num_stages, "window_sizes")
        self.decoder_depths = (
            [int(v) for v in decoder_depths]
            if decoder_depths is not None
            else list(reversed(self.depths[:-1]))
        )
        if len(self.decoder_depths) != max(self.num_stages - 1, 0):
            raise ValueError(
                "decoder_depths length must equal num_stages - 1 "
                f"(got {self.decoder_depths} for {self.num_stages} stages)."
            )

        self.coord_embed = bool(coord_embed)
        self.predict_delta = bool(predict_delta)
        self.full_pos_embed = bool(full_pos_embed)
        self.use_checkpoint = bool(use_checkpoint)
        self.global_modes = global_modes

        stage_dims = [self.hidden_size * (2**stage_idx) for stage_idx in range(self.num_stages)]
        for dim, heads in zip(stage_dims, self.num_heads):
            if dim % heads != 0:
                raise ValueError(
                    f"Stage dim {dim} must be divisible by num_heads {heads}."
                )

        flat_in_channels = self.in_time_window * self.in_channels
        self.patch_embed = PatchEmbed2D(
            in_channels=flat_in_channels,
            embed_dim=self.hidden_size,
            patch_size=self.patch_size,
            norm=patch_norm,
        )
        if self.coord_embed:
            self.coord_patch_embed = PatchEmbed2D(
                in_channels=2,
                embed_dim=self.hidden_size,
                patch_size=self.patch_size,
                norm=False,
            )
        else:
            self.coord_patch_embed = None

        self.pos_embed = None
        if self.full_pos_embed:
            if input_grid_size is None:
                raise ValueError("input_grid_size is required when full_pos_embed=True.")
            input_grid_size = _to_2tuple(input_grid_size)
            patch_grid_size = (
                math.ceil(input_grid_size[0] / self.patch_size[0]),
                math.ceil(input_grid_size[1] / self.patch_size[1]),
            )
            self.pos_embed = nn.Parameter(
                0.02 * torch.randn(1, self.hidden_size, patch_grid_size[0], patch_grid_size[1])
            )

        total_blocks = sum(self.depths) + sum(self.decoder_depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        mixer_period = max(1, int(global_mixer_every))
        global_flags = [
            bool(use_global_mixer) and ((block_idx + 1) % mixer_period == 0)
            for block_idx in range(total_blocks)
        ]

        cursor = 0
        self.encoder_stages = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            depth = self.depths[stage_idx]
            stage_dpr = dpr[cursor : cursor + depth]
            stage_flags = global_flags[cursor : cursor + depth]
            cursor += depth
            self.encoder_stages.append(
                EncoderStage(
                    dim=stage_dims[stage_idx],
                    depth=depth,
                    num_heads=self.num_heads[stage_idx],
                    window_size=self.window_sizes[stage_idx],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rates=stage_dpr,
                    global_flags=stage_flags,
                    global_modes=self.global_modes,
                    layer_scale_init=layer_scale_init,
                    downsample=stage_idx < self.num_stages - 1,
                    use_checkpoint=self.use_checkpoint,
                )
            )

        self.decoder_stages = nn.ModuleList()
        for decoder_idx in range(self.num_stages - 1):
            depth = self.decoder_depths[decoder_idx]
            stage_dpr = dpr[cursor : cursor + depth]
            stage_flags = global_flags[cursor : cursor + depth]
            cursor += depth

            encoder_stage_idx = self.num_stages - 2 - decoder_idx
            in_dim = stage_dims[encoder_stage_idx + 1]
            skip_dim = stage_dims[encoder_stage_idx]
            self.decoder_stages.append(
                DecoderStage(
                    in_dim=in_dim,
                    skip_dim=skip_dim,
                    depth=depth,
                    num_heads=self.num_heads[encoder_stage_idx],
                    window_size=self.window_sizes[encoder_stage_idx],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rates=stage_dpr,
                    global_flags=stage_flags,
                    global_modes=self.global_modes,
                    layer_scale_init=layer_scale_init,
                    use_checkpoint=self.use_checkpoint,
                )
            )

        self.final_expand = PatchExpand2D(
            dim=self.hidden_size,
            out_dim=self.hidden_size,
            scale_factor=self.patch_size,
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.out_channels),
        )

    @staticmethod
    def _normalize_coordinates(coord: torch.Tensor) -> torch.Tensor:
        coord_min = coord.min()
        coord_max = coord.max()
        if float((coord_max - coord_min).abs()) < 1e-6:
            return torch.zeros_like(coord)
        return 2.0 * (coord - coord_min) / (coord_max - coord_min) - 1.0

    def _build_coordinate_grid(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        pos_lst: Optional[Any] = None,
    ) -> torch.Tensor:
        if pos_lst is not None and len(pos_lst) >= 2:
            x_coord = pos_lst[0].squeeze(-1).to(device=device, dtype=torch.float32)
            y_coord = pos_lst[1].squeeze(-1).to(device=device, dtype=torch.float32)

            if x_coord.numel() != height:
                x_min = float(x_coord.min())
                x_max = float(x_coord.max())
                x_coord = torch.linspace(x_min, x_max, height, device=device)
            if y_coord.numel() != width:
                y_min = float(y_coord.min())
                y_max = float(y_coord.max())
                y_coord = torch.linspace(y_min, y_max, width, device=device)

            x_coord = self._normalize_coordinates(x_coord)
            y_coord = self._normalize_coordinates(y_coord)
        else:
            x_coord = torch.linspace(-1.0, 1.0, height, device=device)
            y_coord = torch.linspace(-1.0, 1.0, width, device=device)

        grid_x, grid_y = _meshgrid_ij(x_coord, y_coord)
        coords = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
        return coords.to(dtype=dtype)

    def _prepare_input(
        self, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if u.ndim == 5:
            batch_size, steps, height, width, channels = u.shape
            if steps != self.in_time_window:
                raise ValueError(
                    f"Expected in_time_window={self.in_time_window}, got {steps}."
                )
            if channels != self.in_channels:
                raise ValueError(
                    f"Expected in_channels={self.in_channels}, got {channels}."
                )
            last_frame = u[:, -1]
            x = u.permute(0, 2, 3, 1, 4).reshape(
                batch_size, height, width, steps * channels
            )
        elif u.ndim == 4:
            batch_size, height, width, channels = u.shape
            if self.in_time_window != 1:
                raise ValueError("4D input is only valid when in_time_window=1.")
            if channels != self.in_channels:
                raise ValueError(
                    f"Expected in_channels={self.in_channels}, got {channels}."
                )
            last_frame = u
            x = u
        else:
            raise ValueError(f"Unexpected input shape: {tuple(u.shape)}")

        x = x.permute(0, 3, 1, 2).contiguous()
        return x, last_frame, height, width

    def forward(self, u: torch.Tensor, pos_lst: Optional[Any] = None) -> torch.Tensor:
        x, last_frame, orig_height, orig_width = self._prepare_input(u)
        device = x.device
        dtype = x.dtype

        total_factor_height = self.patch_size[0] * (2 ** (self.num_stages - 1))
        total_factor_width = self.patch_size[1] * (2 ** (self.num_stages - 1))
        pad_height = (total_factor_height - orig_height % total_factor_height) % total_factor_height
        pad_width = (total_factor_width - orig_width % total_factor_width) % total_factor_width
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, pad_width, 0, pad_height))

        padded_height = orig_height + pad_height
        padded_width = orig_width + pad_width

        x = self.patch_embed(x)
        if self.coord_embed and self.coord_patch_embed is not None:
            coords = self._build_coordinate_grid(
                height=padded_height,
                width=padded_width,
                device=device,
                dtype=dtype,
                pos_lst=pos_lst,
            )
            x = x + self.coord_patch_embed(coords)
        if self.full_pos_embed and self.pos_embed is not None:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2:] != x.shape[1:3]:
                pos_embed = F.interpolate(
                    pos_embed,
                    size=(x.shape[1], x.shape[2]),
                    mode="bicubic",
                    align_corners=False,
                )
            x = x + pos_embed.permute(0, 2, 3, 1).contiguous()

        skips: List[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.encoder_stages):
            skip, x = stage(x)
            if stage_idx < self.num_stages - 1:
                skips.append(skip)

        for decoder_stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = decoder_stage(x, skip)

        x = self.final_expand(x)
        x = self.output_norm(x)
        delta = self.output_proj(x)

        if pad_height > 0 or pad_width > 0:
            delta = delta[:, :orig_height, :orig_width, :].contiguous()

        if self.predict_delta and last_frame.shape[-1] == self.out_channels:
            return delta + last_frame[..., : self.out_channels]
        return delta


def build_swin_from_config(config: Any) -> SwinOperatorModel:
    model_cfg = config.model
    in_channels = getattr(model_cfg, "in_channels", getattr(model_cfg, "in_dim", None))
    out_channels = getattr(model_cfg, "out_channels", getattr(model_cfg, "out_dim", None))
    if in_channels is None or out_channels is None:
        raise ValueError(
            "Cannot infer Swin in/out channels from config.model. "
            "Expected one of (in_channels/out_channels) or (in_dim/out_dim)."
        )

    swin_kwargs = {}
    if hasattr(model_cfg, "swin_kwargs") and model_cfg.swin_kwargs is not None:
        swin_kwargs = vars(model_cfg.swin_kwargs)

    if hasattr(model_cfg, "depths"):
        depths = [int(v) for v in model_cfg.depths]
    else:
        depths = [int(getattr(model_cfg, "depth", 6))]

    num_stages = len(depths)
    num_heads = _to_stage_list(getattr(model_cfg, "num_heads", 8), num_stages, "num_heads")

    if hasattr(model_cfg, "window_sizes"):
        window_sizes_value = model_cfg.window_sizes
    else:
        window_ratio = getattr(model_cfg, "window_ratio", None)
        if window_ratio is not None:
            if not hasattr(config, "data") or not hasattr(config.data, "nx") or not hasattr(config.data, "ny"):
                raise ValueError("config.data.nx and config.data.ny are required when using window_ratio.")
            patch_size = _to_2tuple(getattr(model_cfg, "patch_size", 2))
            patch_grid = (
                math.ceil(int(config.data.nx) / patch_size[0]),
                math.ceil(int(config.data.ny) / patch_size[1]),
            )
            window_sizes_value = []
            ratio_h, ratio_w = _to_2tuple(window_ratio)
            for stage_idx in range(num_stages):
                feat_h = max(1, patch_grid[0] // (2**stage_idx))
                feat_w = max(1, patch_grid[1] // (2**stage_idx))
                window_sizes_value.append(
                    (
                        max(1, feat_h // max(1, ratio_h)),
                        max(1, feat_w // max(1, ratio_w)),
                    )
                )
        else:
            window_sizes_value = getattr(model_cfg, "window_size", 8)
    window_sizes = _to_stage_window_list(window_sizes_value, num_stages, "window_sizes")

    decoder_depths = None
    if hasattr(model_cfg, "decoder_depths") and model_cfg.decoder_depths is not None:
        decoder_depths = [int(v) for v in model_cfg.decoder_depths]

    global_modes = _to_2tuple(getattr(model_cfg, "global_modes", (16, 16)))

    return SwinOperatorModel(
        in_channels=in_channels,
        out_channels=out_channels,
        in_time_window=getattr(model_cfg, "in_time_window", 1),
        hidden_size=getattr(model_cfg, "hidden_size", getattr(model_cfg, "dim", 64)),
        patch_size=getattr(model_cfg, "patch_size", 2),
        depths=depths,
        num_heads=num_heads,
        window_sizes=window_sizes,
        decoder_depths=decoder_depths,
        mlp_ratio=getattr(model_cfg, "mlp_ratio", 4.0),
        qkv_bias=getattr(model_cfg, "qkv_bias", True),
        patch_norm=getattr(model_cfg, "patch_norm", True),
        full_pos_embed=getattr(model_cfg, "full_pos_embed", False),
        drop_rate=getattr(model_cfg, "drop_rate", 0.0),
        attn_drop_rate=getattr(model_cfg, "attn_drop_rate", 0.0),
        drop_path_rate=getattr(model_cfg, "drop_path_rate", 0.0),
        coord_embed=getattr(model_cfg, "coord_embed", True),
        predict_delta=getattr(model_cfg, "predict_delta", True),
        use_global_mixer=getattr(model_cfg, "use_global_mixer", False),
        global_mixer_every=getattr(model_cfg, "global_mixer_every", 1),
        global_modes=global_modes,
        layer_scale_init=getattr(model_cfg, "layer_scale_init", 1e-3),
        use_checkpoint=getattr(
            model_cfg,
            "activation_checkpointing",
            getattr(model_cfg, "use_checkpoint", False),
        ),
        input_grid_size=(
            int(config.data.nx),
            int(config.data.ny),
        ) if hasattr(config, "data") and hasattr(config.data, "nx") and hasattr(config.data, "ny") else None,
        **swin_kwargs,
    )

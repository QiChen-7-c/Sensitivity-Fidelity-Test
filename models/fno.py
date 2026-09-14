"""FNO construction and optional activation checkpointing from the original trainer."""
from typing import Optional
import warnings
import torch
from torch.utils.checkpoint import checkpoint
from neuralop.models import FNO

_CHECKPOINT_SUPPORTS_USE_REENTRANT: Optional[bool] = None

def run_activation_checkpoint(
    function,
    *inputs: torch.Tensor,
    use_reentrant: bool,
) -> torch.Tensor:
    global _CHECKPOINT_SUPPORTS_USE_REENTRANT

    if _CHECKPOINT_SUPPORTS_USE_REENTRANT is False:
        return checkpoint(function, *inputs)

    try:
        output = checkpoint(function, *inputs, use_reentrant=use_reentrant)
        _CHECKPOINT_SUPPORTS_USE_REENTRANT = True
        return output
    except TypeError:
        _CHECKPOINT_SUPPORTS_USE_REENTRANT = False
        if not use_reentrant:
            warnings.warn(
                "This PyTorch version does not support checkpoint(..., use_reentrant=...). "
                "Falling back to reentrant activation checkpointing.",
                stacklevel=2,
            )
        return checkpoint(function, *inputs)


class CheckpointedFNO(FNO):
    def __init__(
        self,
        *args,
        activation_checkpointing: bool = False,
        activation_checkpoint_use_reentrant: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_checkpointing = activation_checkpointing
        self.activation_checkpoint_use_reentrant = activation_checkpoint_use_reentrant

    def _forward_fno_block(self, x: torch.Tensor, layer_idx: int, output_shape=None) -> torch.Tensor:
        if (
            not self.activation_checkpointing
            or not self.training
            or not torch.is_grad_enabled()
            or not x.requires_grad
        ):
            return self.fno_blocks(x, layer_idx, output_shape=output_shape)

        def block_forward(block_input: torch.Tensor) -> torch.Tensor:
            return self.fno_blocks(block_input, layer_idx, output_shape=output_shape)

        return run_activation_checkpoint(
            block_forward,
            x,
            use_reentrant=self.activation_checkpoint_use_reentrant,
        )

    def forward(self, x, output_shape=None, **kwargs):
        if kwargs:
            warnings.warn(
                f"FNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self._forward_fno_block(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)
        return x


def build_fno_model(
    config,
    activation_checkpointing: bool = False,
    activation_checkpoint_use_reentrant: bool = False,
) -> FNO:
    n_modes = config.model.n_modes
    if isinstance(n_modes, int):
        n_modes = (n_modes, n_modes)
    else:
        n_modes = tuple(n_modes)

    extra_kwargs = {}
    if hasattr(config.model, "fno_kwargs") and config.model.fno_kwargs is not None:
        extra_kwargs = vars(config.model.fno_kwargs)

    return CheckpointedFNO(
        n_modes=n_modes,
        hidden_channels=config.model.hidden_channels,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        activation_checkpointing=bool(activation_checkpointing),
        activation_checkpoint_use_reentrant=bool(activation_checkpoint_use_reentrant),
        **extra_kwargs,
    )

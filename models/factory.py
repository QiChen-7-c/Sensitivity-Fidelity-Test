"""Single model construction entry point for the four RB models."""
from __future__ import annotations

from typing import Any


def infer_model_type(config: Any, model_type: str = "auto") -> str:
    if model_type != "auto":
        value = model_type.lower()
        if value not in {"ifactformer", "fno", "afno", "swin"}:
            raise ValueError(f"Unsupported model type: {model_type}")
        return value
    model_cfg = config.model
    name = str(getattr(model_cfg, "name", getattr(model_cfg, "model_type", ""))).lower()
    if "swin" in name or hasattr(model_cfg, "depths"):
        return "swin"
    if "afno" in name or hasattr(model_cfg, "num_blocks"):
        return "afno"
    if "fno" in name or hasattr(model_cfg, "n_modes"):
        return "fno"
    return "ifactformer"


def build_fno_model(config, **kwargs):
    from .fno import build_fno_model as _build_fno_model
    return _build_fno_model(config, **kwargs)


def build_model(config, model_type: str = "auto", **kwargs):
    model_type = infer_model_type(config, model_type)
    if model_type == "ifactformer":
        from .ifactformer import Model
        return Model(config.model)
    if model_type == "fno":
        return build_fno_model(config, **kwargs)
    if model_type == "afno":
        from .afno import build_afno_from_config
        return build_afno_from_config(config)
    if model_type == "swin":
        from .swin import build_swin_from_config
        return build_swin_from_config(config)
    raise ValueError(f"Unsupported model type: {model_type}")

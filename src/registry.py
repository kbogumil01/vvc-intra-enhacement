"""
Model registry — register and create models by name.

Usage:
    from registry import MODEL_REGISTRY, create_model, create_lightning_module

    # List available models
    print(MODEL_REGISTRY.keys())

    # Create a model
    model = create_model("dense_base")
    lit = create_lightning_module("dense_large", lr=1e-4)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch.nn as nn


@dataclass
class ModelEntry:
    """Registry entry for a model."""
    config_cls: Type
    model_cls: Type[nn.Module]
    lightning_cls: Type
    default_kwargs: Dict[str, Any]


MODEL_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(
    name: str,
    config_cls: Type,
    model_cls: Type[nn.Module],
    lightning_cls: Type,
    **default_kwargs: Any,
) -> None:
    """Register a model under the given name."""
    MODEL_REGISTRY[name] = ModelEntry(
        config_cls=config_cls,
        model_cls=model_cls,
        lightning_cls=lightning_cls,
        default_kwargs=default_kwargs,
    )


def create_model(name: str, **overrides: Any) -> nn.Module:
    """Create a model by name."""
    entry = MODEL_REGISTRY[name]
    kwargs = {**entry.default_kwargs, **overrides}
    cfg = entry.config_cls(**kwargs)
    return entry.model_cls(cfg)


def create_lightning_module(name: str, lr: float = 1e-4, **overrides: Any):
    """Create a LightningModule by name."""
    entry = MODEL_REGISTRY[name]
    kwargs = {**entry.default_kwargs, **overrides}
    cfg = entry.config_cls(**kwargs)
    return entry.lightning_cls(model_config=cfg, lr=lr, model_name=name)


def get_entry(name: str) -> ModelEntry:
    """Return the registry entry."""
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Return the list of registered names."""
    return sorted(MODEL_REGISTRY.keys())

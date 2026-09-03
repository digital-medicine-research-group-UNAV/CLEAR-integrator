"""Training loop and configuration from the original CLEAR implementation."""

from .._legacy import TrainConfig, _kl_beta, train_cvae

__all__ = ["TrainConfig", "_kl_beta", "train_cvae"]

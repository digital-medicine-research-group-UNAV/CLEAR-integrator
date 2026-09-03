"""CLEAR CVAE module.

The implementation is re-exported from the mechanically preserved legacy module
to keep numerical behavior unchanged during the package refactor.
"""

from .._legacy import ConditionalVAE, VAEConfig

__all__ = ["ConditionalVAE", "VAEConfig"]

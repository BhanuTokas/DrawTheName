"""Loading and embedding text concept banks (spec section 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drawthename.embeddings import ClipLikeBackbone


def load_concept_bank(path: Path) -> list[str]:
    """Reads a newline-delimited concept descriptor file."""
    raise NotImplementedError


def embed_concept_bank(concepts: list[str], backbone: ClipLikeBackbone) -> np.ndarray:
    """Encodes each concept descriptor through the backbone's text encoder."""
    raise NotImplementedError

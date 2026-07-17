"""Loading and embedding text concept banks (spec section 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drawthename.embeddings import ClipLikeBackbone


def load_concept_bank(path: Path) -> list[str]:
    """Reads a newline-delimited concept descriptor file, skipping blank lines and '#' comments."""
    lines = path.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def embed_concept_bank(concepts: list[str], backbone: ClipLikeBackbone) -> np.ndarray:
    """Encodes each concept descriptor through the backbone's text encoder."""
    return backbone.encode_text(concepts)


def center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Mean-centers embeddings, correcting the narrow-cone anisotropy typical
    of CLIP text embeddings for short phrases: concepts cluster tightly
    around a shared direction rather than spreading over the sphere (this
    concept bank's mean embedding has norm ~0.82 despite each concept being
    unit-normed), which otherwise makes retrieval oversensitive to thin
    margins riding on top of a large shared baseline similarity."""
    return embeddings - embeddings.mean(axis=0)

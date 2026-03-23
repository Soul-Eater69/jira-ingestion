"""
Embedding strategy — Section 11 of the architecture spec.

Batches all texts (summary + chunks + metadata) into a single API call.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 512  # OpenAI allows up to 2048 inputs per call


def embed_batch(
    texts: list[str],
    client: Any,
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed a list of texts in batches using the OpenAI embedding API.

    Args:
        texts:      List of strings to embed.
        client:     OpenAI-compatible client.
        model:      Embedding model name.
        batch_size: Maximum number of texts per API call.

    Returns:
        List of embedding vectors (same length as input texts).
    """
    model = model or "text-embedding-3-large"
    all_embeddings: list[list[float]] = []

    num_batches = math.ceil(len(texts) / batch_size)
    for i in range(num_batches):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        # Replace empty strings to avoid API errors
        batch = [t if t.strip() else " " for t in batch]
        try:
            response = client.embeddings.create(input=batch, model=model)
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend(d.embedding for d in sorted_data)
        except Exception as exc:
            logger.error("Embedding failed for batch %d/%d: %s", i + 1, num_batches, exc)
            # Return zero vectors as a graceful fallback
            dim = _get_embedding_dim(model)
            all_embeddings.extend([[0.0] * dim for _ in batch])

    return all_embeddings


def _get_embedding_dim(model: str) -> int:
    """Return the expected dimensionality for common embedding models."""
    dims = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
        "multilingual-e5-large": 1024,
    }
    for key, dim in dims.items():
        if key in model:
            return dim
    return 1536  # safe default

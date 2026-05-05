"""
Embeddings module - Handles sentence embedding model loading and encoding.

Provides a clean interface for embedding generation with lazy loading
and support for multiple embedding models.
"""

import os
import asyncio
from typing import List, Optional

# Model presets
MODELS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "accurate": "sentence-transformers/all-mpnet-base-v2",
}


def get_model_choice() -> str:
    """Get model choice from environment or default to 'fast'."""
    return os.environ.get("AGENT_MEMORY_MODEL", "fast")


def get_model_name(model: Optional[str] = None) -> str:
    """Resolve model name: explicit param > env var > default."""
    if model:
        return model
    choice = get_model_choice()
    return MODELS.get(choice, MODELS["fast"])


def list_models() -> List[str]:
    """List available embedding model names."""
    return list(MODELS.keys())


class EmbeddingEngine:
    """Lazy-loading embedding model wrapper."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_model_name()
        self._model = None

    def _load_model(self):
        """Lazy load the sentence transformer model with optimizations."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                from huggingface_hub import disable_progress_bar
                disable_progress_bar()
            except ImportError:
                # disable_progress_bar not available in this version, ignore
                pass

            # Configure for faster init
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=os.environ.get("TRANSFORMERS_CACHE"),
                device="cpu",
            )
        return self._model

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed
            normalize: Whether to normalize embeddings (default: True)

        Returns:
            List of embedding values
        """
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


class AsyncEmbeddingEngine:
    """Async embedding model wrapper (same implementation, loaded lazily)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_model_name()
        self._model = None

    async def _load_model(self):
        """Lazy load the sentence transformer model with optimizations."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                from huggingface_hub import disable_progress_bar
                disable_progress_bar()
            except ImportError:
                # disable_progress_bar not available in this version, ignore
                pass

            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=os.environ.get("TRANSFORMERS_CACHE"),
                device="cpu",
            )
        return self._model

    async def encode(self, text: str, normalize: bool = True) -> List[float]:
        """Generate embedding for text (async).

        Uses asyncio.to_thread to avoid blocking the event loop.

        Args:
            text: Text to embed
            Whether to normalize embeddings (default: True)

        Returns:
            List of embedding values
        """
        model = await self._load_model()

        def _encode():
            return model.encode(text, normalize_embeddings=normalize)

        embedding = await asyncio.to_thread(_encode)
        return embedding.tolist()

    async def get_dimension(self) -> int:
        """Get embedding dimension."""
        model = await self._load_model()

        def _get_dim():
            return model.get_sentence_embedding_dimension()

        return await asyncio.to_thread(_get_dim)

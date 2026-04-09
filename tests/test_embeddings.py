"""
Unit tests for agent_memory.embeddings module.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_memory.embeddings import (
    MODELS,
    get_model_choice,
    get_model_name,
    list_models,
    EmbeddingEngine,
    AsyncEmbeddingEngine,
)


class TestModelUtils:
    """Tests for model utility functions."""

    def test_list_models_returns_list(self):
        """list_models should return a list of model names."""
        result = list_models()
        assert isinstance(result, list)
        assert "fast" in result
        assert "accurate" in result

    def test_models_contains_expected_keys(self):
        """MODELS dict should contain expected keys."""
        assert "fast" in MODELS
        assert "accurate" in MODELS

    def test_get_model_choice_default(self):
        """get_model_choice should return default when env not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = get_model_choice()
            assert result == "fast"

    def test_get_model_choice_from_env(self):
        """get_model_choice should read from AGENT_MEMORY_MODEL env var."""
        with patch.dict("os.environ", {"AGENT_MEMORY_MODEL": "accurate"}):
            result = get_model_choice()
            assert result == "accurate"

    def test_get_model_name_no_args(self):
        """get_model_name should return default model when no args."""
        with patch.dict("os.environ", {}, clear=True):
            result = get_model_name()
            assert result == MODELS["fast"]

    def test_get_model_name_explicit(self):
        """get_model_name should use explicit param as model name directly."""
        with patch.dict("os.environ", {"AGENT_MEMORY_MODEL": "accurate"}):
            # When explicit param provided, it's used directly (not looked up in MODELS)
            result = get_model_name(MODELS["fast"])
            assert result == MODELS["fast"]

    def test_get_model_name_env_fallback(self):
        """get_model_name should use env when no explicit param."""
        with patch.dict("os.environ", {"AGENT_MEMORY_MODEL": "accurate"}):
            result = get_model_name()
            assert result == MODELS["accurate"]

    def test_get_model_name_unknown_choice(self):
        """get_model_name should fallback to fast for unknown choice."""
        with patch.dict("os.environ", {"AGENT_MEMORY_MODEL": "unknown"}):
            result = get_model_name()
            assert result == MODELS["fast"]


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine class."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_returns_list_of_floats(self, mock_st_class):
        """encode should return a list of floats."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        mock_st_class.return_value = mock_model

        engine = EmbeddingEngine()
        result = engine.encode("test text")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_calls_model_with_normalize(self, mock_st_class):
        """encode should pass normalize parameter to model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1])
        mock_st_class.return_value = mock_model

        engine = EmbeddingEngine()
        engine.encode("test", normalize=True)
        mock_model.encode.assert_called_once_with("test", normalize_embeddings=True)

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_dimension_returns_int(self, mock_st_class):
        """get_dimension should return embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        engine = EmbeddingEngine()
        result = engine.get_dimension()

        assert isinstance(result, int)
        assert result == 384

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_is_lazy_loaded(self, mock_st_class):
        """Model should not be loaded until encode is called."""
        mock_st_class.return_value = MagicMock()

        engine = EmbeddingEngine()
        assert engine._model is None

        engine.encode("test")
        mock_st_class.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_multiple_texts_share_model(self, mock_st_class):
        """Multiple encode calls should use same model instance."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1])
        mock_st_class.return_value = mock_model

        engine = EmbeddingEngine()
        engine.encode("text1")
        engine.encode("text2")

        # Should only be called once (model loaded once)
        assert mock_st_class.call_count == 1


class TestAsyncEmbeddingEngine:
    """Tests for AsyncEmbeddingEngine class."""

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_async_encode_returns_list_of_floats(self, mock_st_class):
        """Async encode should return a list of floats."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        mock_st_class.return_value = mock_model

        engine = AsyncEmbeddingEngine()
        result = await engine.encode("test text")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_async_get_dimension(self, mock_st_class):
        """Async get_dimension should return embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        engine = AsyncEmbeddingEngine()
        result = await engine.get_dimension()

        assert result == 384

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_async_model_is_lazy_loaded(self, mock_st_class):
        """Async model should not be loaded until encode is called."""
        mock_st_class.return_value = MagicMock()

        engine = AsyncEmbeddingEngine()
        assert engine._model is None

        await engine.encode("test")
        mock_st_class.assert_called_once()

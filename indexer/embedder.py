"""
Generate embeddings from text chunks

Supports sentence-transformers (local, free)
Designed to be easily swappable to OpenAI later
"""
from typing import List
from sentence_transformers import SentenceTransformer
from .config import config
from .models import TextChunk


class Embedder:
    """
    Generate embeddings from text

    Currently uses sentence-transformers.
    Can be extended to support OpenAI embeddings later.
    """

    def __init__(self):
        """Initialize the embedding model"""
        if config.embedding_provider == "sentence-transformers":
            print(f"Loading sentence-transformers model: {config.sentence_transformer_model}")
            self.model = SentenceTransformer(config.sentence_transformer_model)
            self.dimension = 384  # all-MiniLM-L6-v2 output dimension
            print(f"✓ Model loaded (dimension: {self.dimension})")
        else:
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a user query (query-time embedding).

        Returns:
            Normalized embedding vector as list of floats
        """
        if config.embedding_provider == "sentence-transformers":
            embedding = self.model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        else:
            raise NotImplementedError(f"Provider {config.embedding_provider} not implemented")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string (document-time embedding).

        Args:
            text: Input text

        Returns:
            Normalized embedding vector as list of floats
        """
        if config.embedding_provider == "sentence-transformers":
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        else:
            raise NotImplementedError(f"Provider {config.embedding_provider} not implemented")

    def embed_chunks(self, chunks: List[TextChunk]) -> List[List[float]]:
        """
        Generate embeddings for multiple chunks (batch processing)

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of normalized embedding vectors (same order as input chunks)
        """
        if config.embedding_provider == "sentence-transformers":
            texts = [chunk.text for chunk in chunks]

            print(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True,
            )

            return [emb.tolist() for emb in embeddings]
        else:
            raise NotImplementedError(f"Provider {config.embedding_provider} not implemented")

    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
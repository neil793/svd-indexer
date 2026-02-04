from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client import models

from indexer.config import config
from indexer.embedder import Embedder


class QdrantStore:
    """
    Qdrant access layer for CMSIS-SVD chunks.

    Assumes the collection is HYBRID configured:
      - named dense vector: "dense" (384-dim cosine)
      - named sparse vector: "bm25" (BM25 sparse vectors via Qdrant inference model)

    Payload schema is unchanged (you already index 'text', 'source_id', etc.).
    """

    DENSE_VECTOR_NAME = "dense"
    BM25_VECTOR_NAME = "bm25"
    BM25_MODEL_NAME = "Qdrant/bm25"
    RRF_K0 = 60

    def __init__(self):
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        self.collection_name = config.collection_name
        self.embedder = Embedder()

    # ---------------------------
    # Helpers
    # ---------------------------

    def embed_query(self, query: str) -> List[float]:
        """
        Query-time embedding. Uses your Embedder.embed_query (normalized).
        """
        return self.embedder.embed_query(query)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convenience function for upsert flows (if you ever use this store for upsert).
        """
        # If you want to keep "store" retrieval-only, you can delete this.
        return [self.embedder.embed_text(t) for t in texts]

    def _rrf_query(self):
        """
        Support both new and old qdrant-client APIs.
        Prefer parameterized RRF (k=60) if available; otherwise fallback to plain RRF.
        """
        if hasattr(models, "RrfQuery") and hasattr(models, "Rrf"):
            return models.RrfQuery(rrf=models.Rrf(k=self.RRF_K0))
        return models.FusionQuery(fusion=models.Fusion.RRF)

    def _to_result(self, pt) -> Dict[str, Any]:
        payload = pt.payload or {}
        return {
            "score": float(pt.score) if pt.score is not None else 0.0,
            "source_id": payload.get("source_id"),
            "type": payload.get("type"), 
            "peripheral": payload.get("peripheral"),
            "register": payload.get("register"),
            "address": payload.get("address"),
            "text": payload.get("text", ""),
            "metadata": payload,  
        }

    # ---------------------------
    # Searches
    # ---------------------------

    def search_vector(
        self,
        query: str,
        top_k: int = 8,
        qfilter: Optional[models.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense-only search (debug baseline).
        """
        qvec = self.embed_query(query)
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
            query_filter=qfilter,
        )
        return [self._to_result(pt) for pt in res.points]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 8,
        vector_k: int = 40,
        bm25_k: int = 40,
        qfilter: Optional[models.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Native hybrid search:
          - dense prefetch on "dense"
          - sparse BM25 prefetch on "bm25"
          - fuse with RRF
        """
        qvec = self.embed_query(query)

        prefetch = [
            models.Prefetch(
                query=qvec,
                using=self.DENSE_VECTOR_NAME,
                limit=vector_k,
                filter=qfilter,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=self.BM25_MODEL_NAME),
                using=self.BM25_VECTOR_NAME,
                limit=bm25_k,
                filter=qfilter,
            ),
        ]

        res = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=self._rrf_query(),
            limit=top_k,
            with_payload=True,
        )

        return [self._to_result(pt) for pt in res.points]

"""Configuration for SVD indexing"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexerConfig:
    """SVD Indexer configuration"""

    # Vector DB - Qdrant (local Docker by default)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")  # usually None for local
    collection_name: str = os.getenv("QDRANT_COLLECTION", "svd_registers")

    # Embedding model
    embedding_provider: str = "sentence-transformers"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # for all-MiniLM-L6-v2
    
    # SVD data directory
    svd_data_dir: str = os.getenv(
        "SVD_DATA_DIR",
        str(Path(__file__).parent.parent / "data" / "svd")
    )

    # Chunking strategy
    include_device: bool = True
    include_peripheral_desc: bool = True
    include_register_desc: bool = True
    include_field_names: bool = True
    include_field_desc: bool = True


config = IndexerConfig()
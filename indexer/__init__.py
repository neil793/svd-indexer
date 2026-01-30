"""
SVD Indexing Pipeline

This package provides tools to parse SVD files, create searchable chunks,
generate embeddings, and index them into a vector database.

Main workflow:
    1. Parse SVD files (parser.py)
    2. Create text chunks (chunker.py)
    3. Generate embeddings (embedder.py)
    4. Index into Qdrant (indexer.py)
"""

from .config import config
from .models import ParsedRegister, TextChunk

__all__ = ['config', 'ParsedRegister', 'TextChunk']
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_URL = "http://localhost:6333"
COLLECTION = "svd_chunks"
VECTOR_SIZE = 384  # change if embedder uses a different dim

client = QdrantClient(url=QDRANT_URL)

existing = [c.name for c in client.get_collections().collections]
if COLLECTION not in existing:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("Created collection:", COLLECTION)
else:
    print("Collection already exists:", COLLECTION)
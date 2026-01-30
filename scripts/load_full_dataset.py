"""
Load the complete SVD dataset into Qdrant.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.parser import parse_svd_file
from indexer.chunker import create_chunk
from indexer.embedder import Embedder
from indexer.indexer import QdrantIndexer
from indexer.config import config


def find_all_svd_files(base_dir: str) -> List[Path]:
    """Recursively find all .svd files."""
    base_path = Path(base_dir)
    svd_files = list(base_path.rglob("*.svd"))
    
    print(f"Found {len(svd_files)} SVD files in {base_dir}")
    
    # Group by vendor
    vendors = {}
    for svd_file in svd_files:
        vendor = svd_file.parent.name
        vendors[vendor] = vendors.get(vendor, 0) + 1
    
    print("\nFiles by vendor:")
    for vendor, count in sorted(vendors.items(), key=lambda x: -x[1]):
        print(f"  {vendor}: {count} files")
    
    return svd_files


def load_full_dataset(svd_dir: str, batch_size: int = 100):
    """Load complete SVD dataset into Qdrant."""
    
    print("=" * 80)
    print("SVD FULL DATASET LOADER")
    print("=" * 80)
    print(f"SVD Directory: {svd_dir}")
    print(f"Qdrant URL: {config.qdrant_url}")
    print(f"Collection: {config.collection_name}")
    print(f"Batch Size: {batch_size}")
    print("=" * 80)
    
    # Find all SVD files
    svd_files = find_all_svd_files(svd_dir)
    
    if not svd_files:
        print(f"\n❌ No SVD files found in {svd_dir}")
        return
    
    # Initialize
    print("\n" + "=" * 80)
    print("INITIALIZING")
    print("=" * 80)
    
    embedder = Embedder()
    indexer = QdrantIndexer()
    
    # Process files
    print("\n" + "=" * 80)
    print("PROCESSING SVD FILES")
    print("=" * 80)
    
    total_chunks = 0
    total_registers = 0
    failed_files = []
    
    for i, svd_file in enumerate(svd_files, 1):
        print(f"\n[{i}/{len(svd_files)}] Processing: {svd_file.name}")
        
        try:
            # Parse
            registers = parse_svd_file(str(svd_file))
            if not registers:
                print(f"  ⚠️  No registers found")
                continue
            
            total_registers += len(registers)
            print(f"  Found {len(registers)} registers")
            
            # Chunk
            chunks = [create_chunk(reg) for reg in registers]
            print(f"  Created {len(chunks)} chunks")
            
            if not chunks:
                continue
            
            # Embed
            print(f"  Embedding...")
            texts = [c.text for c in chunks]
            embeddings = [embedder.embed_text(t) for t in texts]
            
            # Index in batches
            print(f"  Indexing...")
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]
                
                indexer.index_batch(batch_chunks, batch_embeddings)
                print(f"    {batch_end}/{len(chunks)}", end="\r")
            
            print(f"  ✅ Indexed {len(chunks)} chunks")
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_files.append((svd_file.name, str(e)))
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"SVD files: {len(svd_files)}")
    print(f"Registers: {total_registers:,}")
    print(f"Chunks indexed: {total_chunks:,}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for fname, error in failed_files[:10]:
            print(f"  ❌ {fname}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print("\n✅ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd-dir", required=True, help="Directory with SVD files")
    parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    load_full_dataset(args.svd_dir, args.batch_size)
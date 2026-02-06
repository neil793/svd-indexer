"""
Main entry point for SVD indexing pipeline (Qdrant)

Usage:
    # Index a single SVD file
    python -m indexer --svd data/STMicro/STM32F407.svd

    # Index all SVD files in a directory
    python -m indexer --svd-dir data/STMicro/

    # Clear existing index before indexing
    python -m indexer --svd data/STMicro/STM32F407.svd --clear
"""
import argparse
from pathlib import Path
from typing import List

from .parser import parse_multiple_svd_files
from .chunker import create_chunks
from .embedder import Embedder
from .indexer import VectorIndexer
from .deduplicator import deduplicate_registers_exact, get_deduplication_stats


def find_svd_files(directory: str) -> List[str]:
    """
    Recursively find all .svd files in a directory

    Args:
        directory: Path to directory to search

    Returns:
        List of paths to .svd files
    """
    path = Path(directory)
    svd_files = list(path.glob("**/*.svd"))
    return [str(f) for f in sorted(svd_files)]


def main():
    """Run the complete SVD indexing pipeline"""

    parser = argparse.ArgumentParser(
        description="Index SVD files into Qdrant for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index single file
  python -m indexer --svd data/STMicro/STM32F407.svd

  # Index entire vendor directory
  python -m indexer --svd-dir data/STMicro/

  # Clear and reindex
  python -m indexer --svd data/STMicro/STM32F407.svd --clear
        """,
    )

    parser.add_argument("--svd", help="Path to a single SVD file to index")
    parser.add_argument("--svd-dir", help="Directory containing SVD files (searches recursively)")
    parser.add_argument("--clear", action="store_true", help="Clear existing index before indexing")

    args = parser.parse_args()

    # Validate arguments
    if not args.svd and not args.svd_dir:
        parser.error("Must specify either --svd or --svd-dir")
    if args.svd and args.svd_dir:
        parser.error("Cannot specify both --svd and --svd-dir")

    # Determine which SVD files to process
    if args.svd:
        svd_files = [args.svd]
        print(f"Processing single SVD file: {args.svd}")
    else:
        svd_files = find_svd_files(args.svd_dir)
        print(f"Found {len(svd_files)} SVD files in {args.svd_dir}")
        if not svd_files:
            print("No SVD files found!")
            return

    # Print header
    print("\n" + "=" * 70)
    print("SVD INDEXING PIPELINE (QDRANT)")
    print("=" * 70)

    # Step 1: Parse SVD files
    print("\n[1/5] Parsing SVD files...")
    print("-" * 70)
    registers = parse_multiple_svd_files(svd_files)
    print(f"\nOK Parsed {len(registers)} registers total")

    if not registers:
        print("No registers found! Check your SVD files.")
        return

    # Step 2: Deduplicate registers (SAFE: exact duplicates only)
    # IMPORTANT: This should NOT collapse across devices.
    print("\n[2/5] Deduplicating registers (exact duplicates only)...")
    print("-" * 70)

    deduplicated = deduplicate_registers_exact(registers)
    stats = get_deduplication_stats(registers, deduplicated)

    print(f"OK Deduplicated {stats['original_count']} -> {stats['deduplicated_count']} registers")
    print(f"   Reduction: {stats['reduction_count']} registers ({stats['reduction_percent']:.1f}%)")

    registers = deduplicated  # Use deduplicated registers for chunking

    # Step 3: Create searchable chunks
    print("\n[3/5] Creating searchable chunks...")
    print("-" * 70)
    chunks = create_chunks(registers)
    print(f"OK Created {len(chunks)} text chunks")

    if not chunks:
        print("No chunks created! Something went wrong.")
        return

    # Show example chunk
    print("\nExample chunk:")
    print(f"  ID: {chunks[0].id}")
    print(f"  Text: {chunks[0].text[:200]}...")

    # Step 4: Generate embeddings
    print("\n[4/5] Generating embeddings...")
    print("-" * 70)
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    print(f"OK Generated {len(embeddings)} embeddings ({embedder.get_dimension()}-dimensional)")

    if not embeddings:
        print("No embeddings generated! Something went wrong.")
        return

    # Step 5: Index into Qdrant
    print("\n[5/5] Indexing into Qdrant...")
    print("-" * 70)
    indexer = VectorIndexer()

    if args.clear:
        indexer.clear_collection()

    indexer.index_chunks(chunks, embeddings)

    # Show final statistics
    print("\n" + "=" * 70)
    print("INDEXING COMPLETE!")
    print("=" * 70)
    idx_stats = indexer.get_stats()
    print(f"Collection: {idx_stats['collection_name']}")
    print(f"Total documents: {idx_stats['total_documents']}")
    print(f"Qdrant: {idx_stats['qdrant_url']}")
    print("\nYou can now search this data using the search service.")
    print("=" * 70)


if __name__ == "__main__":
    main()

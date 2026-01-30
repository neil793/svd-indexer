# Loading the Full SVD Dataset

## Prerequisites

1. **Qdrant running**:
```bash
   docker run -p 6333:6333 qdrant/qdrant
```

2. **SVD files downloaded**:
   - Clone CMSIS-SVD repository or download SVD files
   - Place in `./data/svd/` or set `SVD_DATA_DIR` environment variable

3. **Dependencies installed**:
```bash
   pip install -r requirements.txt
```

## Loading Steps

### 1. Verify Configuration

Edit `indexer/config.py` and set:
```python
SVD_DATA_DIR = "/path/to/your/svd/files"
```

### 2. Run the Loader
```bash
python scripts/load_full_dataset.py
```

**Options:**
```bash
# Custom SVD directory
python scripts/load_full_dataset.py --svd-dir /path/to/svd

# Larger batch size (faster, more memory)
python scripts/load_full_dataset.py --batch-size 500

# Skip already indexed files
python scripts/load_full_dataset.py --skip-existing
```

### 3. Monitor Progress

The script will show:
- Number of SVD files found
- Files by vendor
- Progress for each file (parsing → chunking → embedding → indexing)
- Final summary with total chunks indexed

**Expected output:**
```
Found 500 SVD files in ./data/svd

Files by vendor:
  STMicro: 200 files
  NXP: 150 files
  ...

[1/500] Processing: STM32F407.svd
  Found 1 device(s)
  Created 1234 register chunks
  Embedding 1234 chunks...
  Indexing in batches of 100...
  ✅ Indexed 1234 chunks

...

SUMMARY
Total SVD files processed: 500
Total devices: 500
Total register chunks indexed: 200,000
Failed files: 0

✅ Dataset loading complete!
```

### 4. Verify Loading
```bash
python -c "
from retrieval.hybrid_retriever import HybridRetriever
r = HybridRetriever(use_reranker=True)
results = r.search('GPIO input register', top_k=5)
for res in results:
    print(f'{res[\"peripheral\"]}/{res[\"register\"]} - {res[\"score\"]:.4f}')
"
```

## Troubleshooting

**Problem: No SVD files found**
- Check `SVD_DATA_DIR` path in config.py
- Verify SVD files have `.svd` extension

**Problem: Out of memory**
- Reduce `--batch-size` (try 50 or 25)
- Process vendors one at a time

**Problem: Qdrant connection error**
- Verify Qdrant is running: `curl http://localhost:6333`
- Check `QDRANT_URL` in config.py

**Problem: Parsing errors**
- Some vendor SVD files may be malformed
- The script will skip failed files and continue
- Check summary for list of failed files
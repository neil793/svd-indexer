from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from qdrant_client import models

from .qdrant_store import QdrantStore
from .reranker import CrossEncoderReranker

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
REG_HINT_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,15}$")
 
# canonical peripheral families triggered by query tokens
PERIPH_TRIGGERS = {
    "uart": ["UART", "USART"],
    "usart": ["USART"],
    "gpio": ["GPIO"],
    "dma": ["DMA"],
    "spi": ["SPI"],
    "i2c": ["I2C"],
    "adc": ["ADC"],
    "dac": ["DAC"],
    "can": ["CAN"],
    "usb": ["USB"],
}


def preprocess_query(query: str) -> Dict[str, Any]:
    raw = query
    clean = query.strip().lower()
    tokens = [t.lower() for t in TOKEN_RE.findall(clean)]

    address_hints = ADDR_RE.findall(raw)

    peripheral_hints = set()
    tokset = set(tokens)
    for trigger, fams in PERIPH_TRIGGERS.items():
        if trigger in tokset:
            peripheral_hints.update(fams)

    register_hints = set()
    for tok in TOKEN_RE.findall(raw):
        if REG_HINT_RE.match(tok):
            register_hints.add(tok)
    
    return {
        "raw": raw,
        "clean": clean,
        "tokens": tokens,
        "peripheral_hints": sorted(peripheral_hints),
        "register_hints": sorted(register_hints),
        "address_hints": address_hints,
    }

def post_process(results: List[Dict[str, Any]], query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply domain boosts AND penalties on top of Qdrant's fused hybrid score.
    This runs BEFORE reranking.
    """
    q_periph = set(query_info["peripheral_hints"])
    q_regs = {r.upper() for r in query_info["register_hints"]}

    out: List[Dict[str, Any]] = []

    for r in results:
        payload = r.get("metadata") or {}
        score = float(r.get("score", 0.0))

        peripheral = (r.get("peripheral") or payload.get("peripheral") or "")
        register = (r.get("register") or payload.get("register") or "")
        chunk_type = payload.get("type", "")

        # Track what boosts/penalties were applied
        peripheral_match = False
        register_match = False
        
        # Boost 1: Exact register name match (very strong signal)
        if q_regs and register.upper() in q_regs:
            score *= 1.4
            register_match = True

        # Penalty 0: Device summaries when querying specific peripheral/register
        if chunk_type == "device_summary" and (q_periph or q_regs):
            score *= 0.7

        rr = dict(r)
        rr["score"] = score

        # Keep debuggability in metadata
        rr["metadata"] = {
            **payload,
            "_debug": {
                "qdrant_hybrid_score": r.get("score"),
                "post_boost_score": score,
                "applied_boosts": {
                    "peripheral_match": peripheral_match,
                    "register_match": register_match,
                },
            },
        }
        out.append(rr)

    # Deterministic tie-break
    out.sort(key=lambda x: (-x["score"], x.get("source_id") or ""))
    return out


def _build_optional_filter(query_info: Dict[str, Any]) -> Optional[models.Filter]:
    """
    Optional: apply strict filtering when query includes an address hint.
    """
    if query_info.get("address_hints"):
        addr = query_info["address_hints"][0]
        return models.Filter(
            must=[models.FieldCondition(key="address", match=models.MatchValue(value=addr))]
        )
    return None


class HybridRetriever:
    """
    Optimal retriever with optional reranking:
      - Qdrant native hybrid (dense + bm25 + RRF)
      - Domain boosting + penalties (before reranking)
      - Optional cross-encoder reranking
      - Post-reranking penalties (to override bad reranker confidence)
    """

    def __init__(self, use_reranker: bool = False):
        """
        Args:
            use_reranker: If True, load cross-encoder for reranking (slower but more accurate)
        """
        self.store = QdrantStore()
        self.use_reranker = use_reranker
        self.reranker = None
        
        if use_reranker:
            self.reranker = CrossEncoderReranker()

    def search(
        self,
        query: str,
        top_k: int = 8,
        vector_k: int = 40,
        bm25_k: int = 40,
        rerank_top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with optional reranking.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            vector_k: Number of candidates from dense vector search
            bm25_k: Number of candidates from BM25 search
            rerank_top_n: Number of top results to rerank (if reranker enabled)
        """
        qi = preprocess_query(query)
        qfilter = _build_optional_filter(qi)

        # Step 1: Qdrant hybrid search + RRF
        base = self.store.search_hybrid(
            query=query,
            top_k=rerank_top_n if self.use_reranker else top_k,
            vector_k=vector_k,
            bm25_k=bm25_k,
            qfilter=qfilter,
        )

        # Step 2: Apply domain-specific boosts and penalties
        boosted = post_process(base, qi)
        
        # Step 3: Optional reranking
        if self.use_reranker and self.reranker:
            reranked = self.reranker.rerank(
                query=query,
                results=boosted[:rerank_top_n],
                top_k=top_k * 2,
                combine_scores=True,
                hybrid_weight=0.5,
                rerank_weight=0.5
            )
            
            return reranked[:top_k]

        
        return boosted[:top_k]
 

# Backwards-compatible function-style entry point
def hybrid_search(
    query: str,
    top_k: int = 8,
    vector_k: int = 40,
    bm25_k: int = 40,
    use_reranker: bool = False,
) -> List[Dict[str, Any]]:
    return HybridRetriever(use_reranker=use_reranker).search(
        query=query,
        top_k=top_k,
        vector_k=vector_k,
        bm25_k=bm25_k,
    )
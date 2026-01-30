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
    "tim": ["TIM"],
    "rcc": ["RCC"],
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

    # Track negative peripheral hints (peripherals to penalize)
    peripheral_penalties = set()
    
    # If query says "DMA" but NOT "USB", penalize USB-related DMA
    if "dma" in tokset and "usb" not in tokset:
        peripheral_penalties.update(["OTG_FS", "OTG_HS", "OTG", "USB"])
    
    # If query says "timer" or "tim", penalize SysTick and Watchdogs
    if any(t in tokset for t in ["timer", "tim"]) and "systick" not in tokset:
        peripheral_penalties.update(["STK", "SYSTICK"])
    
    # If query says "timer" and mentions "prescaler", penalize watchdogs
    if any(t in tokset for t in ["timer", "tim", "prescaler"]):
        peripheral_penalties.update(["WWDG", "IWDG"])
    
    # If query mentions I2C, penalize USB I2C controllers
    if "i2c" in tokset and "usb" not in tokset:
        peripheral_penalties.update(["OTG_FS", "OTG_HS", "USB"])

    return {
        "raw": raw,
        "clean": clean,
        "tokens": tokens,
        "peripheral_hints": sorted(peripheral_hints),
        "register_hints": sorted(register_hints),
        "address_hints": address_hints,
        "peripheral_penalties": sorted(peripheral_penalties),
    }


def _periph_family(periph: str) -> str:
    p = (periph or "").upper()
    for fam in ["USART", "UART", "GPIO", "SPI", "I2C", "TIM", "RCC", "ADC", "DAC", "CAN", "USB", "OTG"]:
        if p.startswith(fam):
            return fam
    return p


def post_process(results: List[Dict[str, Any]], query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply domain boosts AND penalties on top of Qdrant's fused hybrid score.
    This runs BEFORE reranking.
    """
    q_periph = set(query_info["peripheral_hints"])
    q_regs = {r.upper() for r in query_info["register_hints"]}
    q_penalties = set(query_info.get("peripheral_penalties", []))

    out: List[Dict[str, Any]] = []

    for r in results:
        payload = r.get("metadata") or {}
        score = float(r.get("score", 0.0))

        peripheral = (r.get("peripheral") or payload.get("peripheral") or "")
        register = (r.get("register") or payload.get("register") or "")
        peripheral_upper = peripheral.upper()

        # Track what boosts/penalties were applied
        peripheral_match = False
        register_match = False
        peripheral_penalty = False

        # Boost 1: Peripheral family match
        if q_periph:
            fam = _periph_family(peripheral)
            if fam in q_periph:
                score *= 1.5
                peripheral_match = True
        
        # Boost 2: Exact register name match (very strong signal)
        if q_regs and register.upper() in q_regs:
            score *= 1.4
            register_match = True

        # Penalty 1: Wrong peripheral family
        if q_penalties:
            for penalty_periph in q_penalties:
                if peripheral_upper.startswith(penalty_periph):
                    score *= 0.5
                    peripheral_penalty = True
                    break

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
                    "peripheral_penalty": peripheral_penalty,
                },
                "peripheral_penalties_active": sorted(q_penalties) if q_penalties else [],
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

    def _apply_post_rerank_penalties(
        self, 
        results: List[Dict[str, Any]], 
        query_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply STRONG penalties after reranking to override bad reranker confidence.
        
        This is critical because the reranker can be 99%+ confident on wrong answers
        due to semantic similarity (e.g., STK/VAL for "timer counter").
        """
        q_penalties = set(query_info.get("peripheral_penalties", []))
        
        if not q_penalties:
            return results
        
        for r in results:
            peripheral = (r.get("peripheral") or "").upper()
            
            # VERY STRONG penalty after reranking (must override 99%+ confidence)
            for penalty_periph in q_penalties:
                if peripheral.startswith(penalty_periph):
                    # Save original score for debugging
                    original_score = r["score"]
                    
                    # Apply aggressive penalty
                    r["score"] *= 0.1  # 90% reduction
                    
                    # Update debug info
                    metadata = r.get("metadata", {})
                    debug = metadata.get("_debug", {})
                    debug["post_rerank_penalty"] = True
                    debug["pre_penalty_score"] = original_score
                    debug["penalty_applied"] = penalty_periph
                    metadata["_debug"] = debug
                    r["metadata"] = metadata
                    break
        
        # Re-sort after penalties
        results.sort(key=lambda x: (-x["score"], x.get("source_id") or ""))
        return results

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
            # Rerank top N results
            reranked = self.reranker.rerank(
                query=query,
                results=boosted[:rerank_top_n],
                top_k=top_k * 2,  # Get extra results before final penalties
                combine_scores=True,
                hybrid_weight=0.5,  # 50/50 balance
                rerank_weight=0.5
            )
            
            # Step 4: Apply post-reranking penalties to override bad confidence
            final = self._apply_post_rerank_penalties(reranked, qi)
            return final[:top_k]
        
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
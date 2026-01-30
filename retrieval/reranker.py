"""
Cross-encoder reranking for improved result quality.

Cross-encoders jointly encode query+document pairs for more accurate relevance scoring
than bi-encoders (which encode query and document separately).
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Rerank search results using a cross-encoder model.
    
    Cross-encoders are slower but more accurate than bi-encoders because they
    can attend to interactions between query and document tokens.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder model.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, 80MB)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (better, 120MB)
                - "cross-encoder/ms-marco-TinyBERT-L-2-v2" (fastest, 40MB)
        """
        print(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"✓ Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = None,
        combine_scores: bool = True,
        hybrid_weight: float = 0.3,
        rerank_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results from hybrid retriever
            top_k: Number of top results to return (None = return all)
            combine_scores: If True, combine hybrid score with rerank score
            hybrid_weight: Weight for original hybrid score (if combine_scores=True)
            rerank_weight: Weight for reranker score (if combine_scores=True)
        
        Returns:
            Reranked results with updated scores
        """
        if not results:
            return results
        
        # Build query-document pairs for cross-encoder
        pairs = []
        for result in results:
            # Use the full text for reranking
            doc_text = result.get("text", "")
            
            # Fallback if text is not in result
            if not doc_text:
                metadata = result.get("metadata", {})
                doc_text = metadata.get("text", "")
            
            # Further fallback: construct from available fields
            if not doc_text:
                peripheral = result.get("peripheral", "")
                register = result.get("register", "")
                doc_text = f"{peripheral} {register}"
            
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores (batch processing)
        rerank_scores = self.model.predict(pairs)
        
        # Normalize rerank scores to 0-1 range using sigmoid
        rerank_scores_normalized = 1 / (1 + np.exp(-np.array(rerank_scores)))
        
        # Combine scores or replace
        reranked_results = []
        for i, result in enumerate(results):
            result_copy = dict(result)
            original_score = float(result.get("score", 0.0))
            rerank_score = float(rerank_scores_normalized[i])
            
            if combine_scores:
                # Weighted combination of hybrid and rerank scores
                final_score = (
                    hybrid_weight * original_score + 
                    rerank_weight * rerank_score
                )
            else:
                # Use only rerank score
                final_score = rerank_score
            
            result_copy["score"] = final_score
            
            # Update debug info
            metadata = result_copy.get("metadata", {})
            debug = metadata.get("_debug", {})
            debug["rerank_score"] = rerank_score
            debug["rerank_score_raw"] = float(rerank_scores[i])
            debug["hybrid_score"] = original_score
            debug["combined"] = combine_scores
            metadata["_debug"] = debug
            result_copy["metadata"] = metadata
            
            reranked_results.append(result_copy)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: -x["score"])
        
        # Return top-k if specified
        if top_k is not None:
            return reranked_results[:top_k]
        
        return reranked_results
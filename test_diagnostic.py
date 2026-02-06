#!/usr/bin/env python3
"""
Diagnostic test suite to investigate failures in EXTI, USART, and SPI queries.

This test:
1. Searches for failing queries
2. Shows what chunks are actually being returned
3. Analyzes chunk content to understand why correct answers are missing
4. Compares inventory vs single-register query behavior
"""

import sys
from typing import List, Dict, Any
from retrieval.hybrid_retriever import HybridRetriever


# Focused test cases - only the problem areas
DIAGNOSTIC_TESTS = [
    # === INVENTORY QUERIES (should return device_summary chunks) ===
    {
        "name": "STM32F030 timers inventory",
        "query": "On the STM32F030, what timers are available?",
        "expected_chunk_type": "device_summary",
        "expected_peripheral": "TIM",
        "expected_in_text": ["TIM1", "TIM3", "TIM14", "TIM16", "TIM17"],
        "category": "inventory"
    },
    {
        "name": "STM32F429 USART inventory",
        "query": "On the STM32F429, what USART/UART peripherals exist?",
        "expected_chunk_type": "device_summary",
        "expected_peripheral": "USART",
        "expected_in_text": ["USART1", "USART2", "USART3", "UART4", "UART5"],
        "category": "inventory"
    },
    {
        "name": "STM32F746 SPI inventory",
        "query": "On the STM32F746, what SPI peripherals are available?",
        "expected_chunk_type": "device_summary",
        "expected_peripheral": "SPI",
        "expected_in_text": ["SPI1", "SPI2", "SPI3", "SPI4"],
        "category": "inventory"
    },
    
    # === REGISTER-SPECIFIC QUERIES (should return peripheral_summary or peripheral_detail) ===
    {
        "name": "USART TXE status bit",
        "query": "Which USART register on STM32F103 indicates TXE (transmit data register empty)?",
        "expected_chunk_type": "peripheral_summary",  # or peripheral_detail
        "expected_peripheral": "USART",
        "expected_in_text": ["SR", "USART_SR"],
        "category": "register"
    },
    {
        "name": "SPI TXE status bit",
        "query": "Which SPI status register bit indicates the transmit buffer is empty on STM32F407?",
        "expected_chunk_type": "peripheral_summary",
        "expected_peripheral": "SPI",
        "expected_in_text": ["SR", "TXE", "SPI_SR"],
        "category": "register"
    },
    {
        "name": "EXTI IMR enable",
        "query": "Which EXTI register enables an interrupt line on STM32F407?",
        "expected_chunk_type": "peripheral_summary",
        "expected_peripheral": "EXTI",
        "expected_in_text": ["IMR", "EXTI_IMR"],
        "category": "register"
    },
    {
        "name": "EXTI RTSR rising edge",
        "query": "Which EXTI register selects rising edge trigger on STM32F103?",
        "expected_chunk_type": "peripheral_summary",
        "expected_peripheral": "EXTI",
        "expected_in_text": ["RTSR", "EXTI_RTSR"],
        "category": "register"
    },
    {
        "name": "DMA NDTR counter",
        "query": "Which DMA register on STM32F411 stores the number of data items to transfer?",
        "expected_chunk_type": "peripheral_summary",
        "expected_peripheral": "DMA",
        "expected_in_text": ["NDTR", "CNDTR"],
        "category": "register"
    },
    {
        "name": "GPIO CRL/CRH config",
        "query": "Which GPIO register on STM32F103 configures a pin as input/output/alt-function (mode/cnf)?",
        "expected_chunk_type": "peripheral_summary",
        "expected_peripheral": "GPIO",
        "expected_in_text": ["CRL", "CRH", "GPIO_CRL", "GPIO_CRH"],
        "category": "register"
    },
]


def analyze_chunk(chunk: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single chunk to understand why it did/didn't match.
    
    Returns dict with analysis results.
    """
    text = chunk.get("text", "").upper()
    metadata = chunk.get("metadata", {})
    chunk_type = metadata.get("type", "unknown")
    peripheral = chunk.get("peripheral", "").upper()
    
    # Check what we expected vs what we got
    analysis = {
        "chunk_type": chunk_type,
        "chunk_type_match": chunk_type == test["expected_chunk_type"],
        "peripheral": peripheral,
        "peripheral_match": test["expected_peripheral"].upper() in peripheral,
        "expected_tokens_found": [],
        "expected_tokens_missing": [],
        "text_length": len(text),
        "register_count": metadata.get("register_count", 0),
        "registers_in_metadata": metadata.get("registers", []),
    }
    
    # Check for expected tokens
    for token in test["expected_in_text"]:
        if token.upper() in text:
            analysis["expected_tokens_found"].append(token)
        else:
            analysis["expected_tokens_missing"].append(token)
    
    # Special checks for inventory queries
    if test["category"] == "inventory":
        analysis["is_device_summary"] = chunk_type == "device_summary"
        analysis["has_multiple_instances"] = len(analysis["expected_tokens_found"]) >= 2
    
    # Special checks for register queries
    if test["category"] == "register":
        analysis["has_register_names"] = bool(analysis["registers_in_metadata"])
        analysis["has_field_names"] = "field_names" in metadata and bool(metadata["field_names"])
    
    return analysis


def print_chunk_detail(rank: int, chunk: Dict[str, Any], analysis: Dict[str, Any]):
    """Pretty print chunk details."""
    print(f"\n   {'👈' if rank == 1 else '  '} Rank #{rank}")
    print(f"      Score: {chunk['score']:.6f}")
    print(f"      Type: {analysis['chunk_type']} {'✅' if analysis['chunk_type_match'] else '❌'}")
    print(f"      Peripheral: {analysis['peripheral']} {'✅' if analysis['peripheral_match'] else '❌'}")
    
    if analysis["expected_tokens_found"]:
        print(f"      ✅ Found tokens: {', '.join(analysis['expected_tokens_found'])}")
    if analysis["expected_tokens_missing"]:
        print(f"      ❌ Missing tokens: {', '.join(analysis['expected_tokens_missing'])}")
    
    print(f"      Text length: {analysis['text_length']} chars")
    
    if analysis.get("register_count"):
        print(f"      Registers: {analysis['register_count']} ({', '.join(analysis['registers_in_metadata'][:5])}...)")
    
    # Show text preview
    text_preview = chunk.get("text", "")[:150].replace("\n", " ")
    print(f"      Preview: {text_preview}...")
    
    # Show debug info if available
    debug = chunk.get("metadata", {}).get("_debug", {})
    if debug:
        print(f"      Debug:")
        if "hybrid_score" in debug:
            print(f"         Hybrid: {debug['hybrid_score']:.6f}")
        if "rerank_score" in debug:
            print(f"         Rerank: {debug['rerank_score']:.6f}")
        if debug.get("applied_boosts", {}).get("peripheral_match"):
            print(f"         ✓ Peripheral boost applied")


def run_diagnostic_tests(use_reranker: bool = False):
    """Run diagnostic tests and provide detailed analysis."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST SUITE - PROBLEM AREA INVESTIGATION")
    print("=" * 80)
    if use_reranker:
        print("🔥 RERANKING ENABLED")
    print()
    
    retriever = HybridRetriever(use_reranker=use_reranker)
    
    # Track results by category
    inventory_results = {"passed": 0, "failed": 0}
    register_results = {"passed": 0, "failed": 0}
    
    for test in DIAGNOSTIC_TESTS:
        print("\n" + "=" * 80)
        print(f"TEST: {test['name']}")
        print("=" * 80)
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected_chunk_type']} chunk with {test['expected_peripheral']}")
        print(f"Should contain: {', '.join(test['expected_in_text'])}")
        print()
        
        # Search with retriever
        results = retriever.search(test["query"], top_k=5)
        
        if not results:
            print("❌ NO RESULTS RETURNED")
            if test["category"] == "inventory":
                inventory_results["failed"] += 1
            else:
                register_results["failed"] += 1
            continue
        
        # Analyze top 5 results
        analyses = [analyze_chunk(r, test) for r in results[:5]]
        
        # Check if any result is correct
        top_result_correct = (
            analyses[0]["chunk_type_match"] and
            analyses[0]["peripheral_match"] and
            len(analyses[0]["expected_tokens_found"]) > 0
        )
        
        any_in_top5_correct = any(
            a["chunk_type_match"] and 
            a["peripheral_match"] and 
            len(a["expected_tokens_found"]) > 0
            for a in analyses
        )
        
        # Print result
        if top_result_correct:
            print("✅ TOP RESULT CORRECT")
            if test["category"] == "inventory":
                inventory_results["passed"] += 1
            else:
                register_results["passed"] += 1
        elif any_in_top5_correct:
            correct_rank = next(
                i+1 for i, a in enumerate(analyses)
                if a["chunk_type_match"] and a["peripheral_match"] and len(a["expected_tokens_found"]) > 0
            )
            print(f"⚠️  CORRECT ANSWER AT RANK #{correct_rank}")
            if test["category"] == "inventory":
                inventory_results["failed"] += 1
            else:
                register_results["failed"] += 1
        else:
            print("❌ CORRECT ANSWER NOT IN TOP 5")
            if test["category"] == "inventory":
                inventory_results["failed"] += 1
            else:
                register_results["failed"] += 1
        
        # Show all top 5 results with analysis
        print("\nTop 5 Results:")
        for i, (result, analysis) in enumerate(zip(results[:5], analyses), 1):
            print_chunk_detail(i, result, analysis)
        
        print()
    
    # Final summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    inv_total = inventory_results["passed"] + inventory_results["failed"]
    inv_rate = (inventory_results["passed"] / inv_total * 100) if inv_total > 0 else 0
    print(f"\nInventory Queries (device listings):")
    print(f"  {inventory_results['passed']}/{inv_total} passed ({inv_rate:.1f}%)")
    print(f"  Expected: device_summary chunks ranking first")
    
    reg_total = register_results["passed"] + register_results["failed"]
    reg_rate = (register_results["passed"] / reg_total * 100) if reg_total > 0 else 0
    print(f"\nRegister Queries (specific register names):")
    print(f"  {register_results['passed']}/{reg_total} passed ({reg_rate:.1f}%)")
    print(f"  Expected: peripheral_summary chunks with register names")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if inv_rate < 80:
        print("\n⚠️  Inventory queries failing:")
        print("   - device_summary chunks not ranking high enough")
        print("   - Reranker may be boosting single peripheral instances")
        print("   - Need stronger boost for device_summary on 'what/which/list' queries")
    
    if reg_rate < 80:
        print("\n⚠️  Register queries failing:")
        print("   - Check if peripheral_summary chunks contain register names")
        print("   - Check if field names (TXE, IMR, etc.) are indexed")
        print("   - May need register-level detail chunks for critical registers")
    
    print()


if __name__ == "__main__":
    use_reranker = "--rerank" in sys.argv
    run_diagnostic_tests(use_reranker=use_reranker)
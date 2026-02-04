"""
Test retrieval for STM32F407 device (191 chunks in Qdrant)
Tests if peripheral-level chunking + BGE-M3 embeddings work well
"""

import sys
from typing import Dict, Any
from retrieval.hybrid_retriever import HybridRetriever


# Test cases - STM32F407 ONLY
TEST_CASES = [
    {
        "query": "Which USART register on STM32F407 is used to WRITE transmit data?",
        "peripheral": "USART",
        "registers": ["DR", "TDR", "TXDR"],
        "category": "USART"
    },
    {
        "query": "Which GPIO register on STM32F407 WRITES the output pin state?",
        "peripheral": "GPIO",
        "registers": ["ODR", "BSRR"],
        "category": "GPIO"
    },
    {
        "query": "Which ADC register on STM32F407 READS the conversion result?",
        "peripheral": "ADC",
        "registers": ["DR", "ADC_DR"],
        "category": "ADC"
    },
    {
        "query": "Which IWDG register on STM32F407 reloads the watchdog counter?",
        "peripheral": "IWDG",
        "registers": ["KR", "IWDG_KR"],
        "category": "IWDG"
    },
    {
        "query": "Which timer register on STM32F407 sets the PWM duty cycle?",
        "peripheral": "TIM",
        "registers": ["CCR", "CCR1", "CCR2", "CCR3", "CCR4"],
        "category": "Timer"
    },
    {
        "query": "Which RCC register on STM32F407 enables peripheral clocks?",
        "peripheral": "RCC",
        "registers": ["ENR", "AHB1ENR", "AHB2ENR", "APB1ENR", "APB2ENR"],
        "category": "RCC"
    },
    {
        "query": "Which SPI register on STM32F407 enables the peripheral?",
        "peripheral": "SPI",
        "registers": ["CR1", "SPI_CR1"],
        "category": "SPI"
    },
    {
        "query": "Which DMA register on STM32F407 stores the number of data items to transfer?",
        "peripheral": "DMA",
        "registers": ["CNDTR", "NDTR"],
        "exclude_peripherals": ["OTG", "USB"],
        "category": "DMA"
    },
]


def check_result(result: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """Check if result matches test expectations for peripheral-level chunks."""
    peripheral = (result.get("peripheral") or "").upper()
    
    # For peripheral-level chunks, we just check if the right peripheral was found
    expected_periph = test["peripheral"].upper()
    if expected_periph not in peripheral:
        return False
    
    # Check for excluded peripherals
    if "exclude_peripherals" in test:
        for excluded in test["exclude_peripherals"]:
            if excluded.upper() in peripheral:
                return False
    
    # For peripheral chunks, check if the expected register is mentioned in the text or registers list
    metadata = result.get("metadata", {})
    registers_list = metadata.get("registers", [])
    text = result.get("text", "")
    
    # Check if any of the acceptable registers are in the peripheral's register list or text
    for acceptable_reg in test["registers"]:
        reg_upper = acceptable_reg.upper()
        # Check in registers list
        if any(reg_upper in r.upper() for r in registers_list):
            return True
        # Check in text content
        if reg_upper in text.upper():
            return True
    
    return False


def run_tests(use_reranker: bool = False, verbose: bool = False):
    """Run all test cases and report results."""
    print("\n" + "=" * 80)
    print("STM32F407 RETRIEVAL TEST (191 chunks)")
    if use_reranker:
        print("🔥 RERANKING ENABLED")
    print("=" * 80)
    
    retriever = HybridRetriever(use_reranker=use_reranker)
    
    passed = 0
    failed = 0
    results_by_category = {}
    
    for test in TEST_CASES:
        category = test["category"]
        
        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "failed": 0}
        
        results = retriever.search(test["query"], top_k=10)
        
        if not results:
            failed += 1
            results_by_category[category]["failed"] += 1
            print(f"\n❌ FAIL: {test['query'][:60]}...")
            print(f"   No results returned!")
            continue
        
        top_result = results[0]
        is_correct = check_result(top_result, test)
        
        if is_correct:
            passed += 1
            results_by_category[category]["passed"] += 1
            if verbose:
                print(f"✅ {test['query'][:60]}...")
                print(f"   Got: {top_result['peripheral']} (contains {test['registers'][0]})")
        else:
            failed += 1
            results_by_category[category]["failed"] += 1
            
            # Find if correct answer is in top 5
            correct_rank = None
            for i, r in enumerate(results[:5], 1):
                if check_result(r, test):
                    correct_rank = i
                    break
            
            rank_info = f" (correct at #{correct_rank})" if correct_rank else ""
            print(f"\n{'=' * 80}")
            print(f"❌ FAIL: {test['query']}")
            print(f"🎯 Expected: {test['peripheral']} peripheral with {test['registers'][0]} register")
            print(f"{'=' * 80}")
            print(f"   Got: {top_result['peripheral']}")
            print(f"   Score: {top_result['score']:.6f}")
            
            # Check if the register is in the chunk
            metadata = top_result.get("metadata", {})
            registers_in_chunk = metadata.get("registers", [])
            has_register = any(test["registers"][0].upper() in r.upper() for r in registers_in_chunk)
            if has_register:
                print(f"   ℹ️  Register {test['registers'][0]} IS in this peripheral chunk")
            else:
                print(f"   ⚠️  Register {test['registers'][0]} NOT FOUND in this peripheral chunk")
            
            # Show top 5 results
            print(f"\n   Top 5 results:")
            for i, r in enumerate(results[:5], 1):
                marker = "👈" if i == 1 else "  "
                is_match = check_result(r, test)
                match_symbol = "✅" if is_match else ""
                periph_name = r.get('peripheral', 'Unknown')
                print(f"   {marker} {i}. {periph_name} (score: {r['score']:.6f}) {match_symbol}")
                
                # Show debug info if available
                debug = r.get("metadata", {}).get("_debug", {})
                if debug:
                    if use_reranker and "rerank_score" in debug:
                        print(f"       Hybrid: {debug.get('hybrid_score', 0):.6f} | Rerank: {debug.get('rerank_score', 0):.6f}")
                    else:
                        print(f"       Qdrant: {debug.get('qdrant_hybrid_score', 0):.6f}")
            
            print()
    
    # Print summary by peripheral category
    print("\n" + "=" * 80)
    print("SUMMARY BY PERIPHERAL")
    print("=" * 80)
    
    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        total = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / total * 100) if total > 0 else 0
        
        status = "✅" if rate == 100 else "⚠️" if rate >= 50 else "❌"
        print(f"{status} {category:12} {stats['passed']:2}/{total:2} ({rate:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("OVERALL")
    print("=" * 80)
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Total: {total} | ✅ {passed} | ❌ {failed} | Rate: {success_rate:.1f}%")
    
    if use_reranker:
        print("\n💡 Reranker was used to rescore top results")
    
    print()
    return success_rate >= 70.0


if __name__ == "__main__":
    use_reranker = "--rerank" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    success = run_tests(use_reranker=use_reranker, verbose=verbose)
    exit(0 if success else 1)
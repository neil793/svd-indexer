# test_realistic_queries.py

"""
Realistic developer question test suite for SVD register retrieval.
Tests common embedded development queries against the hybrid search system.
"""

import sys
import os
from typing import List, Dict, Any, Set
from retrieval.hybrid_retriever import HybridRetriever

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Test cases - STM32F4 Family only (10 tests)
TEST_CASES = [
    {
        "query": "On the STM32F030, what timers are available?",
        "peripheral": "TIM",
        "registers": ["TIM1", "TIM3", "TIM14", "TIM16", "TIM17"],
        "device_family": "STM32F0",
        "category": "Timer"
    },
    {
        "query": "Which USART register on STM32F407 is used to WRITE transmit data?",
        "peripheral": "USART",
        "registers": ["DR", "TDR", "TXDR"],
        "device_family": "STM32F4",
        "category": "USART"
    },
    {
        "query": "Which SPI CONTROL register on STM32F411 enables the peripheral?",
        "peripheral": "SPI",
        "registers": ["CR1", "SPI_CR1"],
        "device_family": "STM32F4",
        "category": "SPI"
    },
    {
        "query": "Which I2C STATUS register on STM32F429 indicates ACK or NACK?",
        "peripheral": "I2C",
        "registers": ["SR", "ISR", "I2C_ISR", "SR1", "SR2"],
        "device_family": "STM32F4",
        "category": "I2C"
    },
    {
        "query": "Which GPIO register on STM32F407 WRITES the output pin state?",
        "peripheral": "GPIO",
        "registers": ["ODR", "BSRR"],
        "device_family": "STM32F4",
        "category": "GPIO"
    },
    {
        "query": "Which timer register on STM32F429 sets capture/compare PWM duty cycle values?",
        "peripheral": "TIM",
        "registers": ["CCR", "CCR1", "CCR2", "CCR3", "CCR4"],
        "device_family": "STM32F4",
        "category": "Timer"
    },
    {
        "query": "Which ADC register on STM32F407 READS the conversion result?",
        "peripheral": "ADC",
        "registers": ["DR", "ADC_DR"],
        "device_family": "STM32F4",
        "category": "ADC"
    },
    {
        "query": "Which DMA register on STM32F411 stores the number of data items to transfer?",
        "peripheral": "DMA",
        "registers": ["CNDTR", "NDTR"],
        "exclude_peripherals": ["OTG", "USB"],
        "device_family": "STM32F4",
        "category": "DMA"
    },
    {
        "query": "Which RCC register on STM32F429 ENABLES peripheral clocks?",
        "peripheral": "RCC",
        "registers": ["ENR", "AHB1ENR", "AHB2ENR", "APB1ENR", "APB2ENR"],
        "device_family": "STM32F4",
        "category": "RCC"
    },
    {
        "query": "Which IWDG register on STM32F407 reloads the watchdog counter?",
        "peripheral": "IWDG",
        "registers": ["KR", "IWDG_KR"],
        "device_family": "STM32F4",
        "category": "IWDG"
    },
    {
        "query": "Which RTC register on STM32F429 stores the current TIME?",
        "peripheral": "RTC",
        "registers": ["TR", "RTC_TR", "TSTR"],
        "device_family": "STM32F4",
        "category": "RTC"
    },
]


def check_result(result: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """Check if result matches test expectations."""
    peripheral = (result.get("peripheral") or "").upper()
    register = (result.get("register") or "").upper()
    
    expected_periph = test["peripheral"].upper()
    if expected_periph not in peripheral:
        return False
    
    if "exclude_peripherals" in test:
        for excluded in test["exclude_peripherals"]:
            if excluded.upper() in peripheral:
                return False
    
    for acceptable_reg in test["registers"]:
        if acceptable_reg.upper() in register:
            return True
    
    return False


def query_llm_for_answer(query: str, top_results: List[Dict[str, Any]], openai_client) -> str:
    """
    Use OpenAI to answer the user's question based on retrieval results.
    
    Args:
        query: User's original question
        top_results: Top search results from retrieval system
        openai_client: OpenAI client instance
    
    Returns:
        LLM's answer
    """
    # Format results as context
    context_parts = []
    for i, result in enumerate(top_results[:5], 1):
        peripheral = result.get('peripheral', 'Unknown')
        register = result.get('register', 'Unknown')
        description = result.get('description', 'No description available')
        
        context_parts.append(
            f"Result {i}: {peripheral}/{register}\n"
            f"Description: {description}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Create prompt
    system_prompt = """You are an expert embedded systems engineer helping with STM32 microcontroller register questions.
Given search results from an SVD database, provide a clear, concise answer to the user's question.
Focus on identifying the correct register name and explaining its purpose briefly."""
    
    user_prompt = f"""Question: {query}

Search Results:
{context}

Based on these search results, answer the user's question. Be specific about which register to use and why."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying LLM: {str(e)}"


def run_tests(use_reranker: bool = False, verbose: bool = False, use_llm: bool = False):
    """Run all test cases and report results."""
    print("\n" + "=" * 80)
    print("SVD REGISTER RETRIEVAL TEST SUITE - STM32F4 FAMILY (10 TESTS)")
    if use_reranker:
        print("🔥 RERANKING ENABLED")
    if use_llm:
        print("🤖 LLM ANSWER GENERATION ENABLED")
    print("=" * 80)
    
    # Initialize OpenAI if requested
    openai_client = None
    if use_llm:
        if not OPENAI_AVAILABLE:
            print("⚠️  OpenAI not installed. Install with: pip install openai")
            print("    Proceeding without LLM...\n")
            use_llm = False
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not set. Proceeding without LLM...\n")
                use_llm = False
            else:
                openai_client = OpenAI(api_key=api_key)
                print("✓ OpenAI initialized\n")
    
    retriever = HybridRetriever(use_reranker=use_reranker)
    
    results_by_category = {}
    passed = 0
    failed = 0
    
    for test in TEST_CASES:
        category = test["category"]
        
        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "failed": 0}
        
        results = retriever.search(test["query"], top_k=10)
        
        if not results:
            failed += 1
            results_by_category[category]["failed"] += 1
            if verbose:
                print(f"❌ {test['query'][:60]}... | No results")
            continue
        
        top_result = results[0]
        is_correct = check_result(top_result, test)
        
        if is_correct:
            passed += 1
            results_by_category[category]["passed"] += 1
            if verbose:
                print(f"✅ {test['query'][:60]}...")
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
            print(f"📍 Expected: {test['peripheral']}/{test['registers'][0]}")
            print(f"{'=' * 80}")
            print(f"   Got: {top_result['peripheral']}/{top_result['register']}{rank_info}")
            print(f"   Score: {top_result['score']:.6f}")
            
            # Show top 5 results
            print(f"\n   Top 5 results:")
            for i, r in enumerate(results[:5], 1):
                marker = "👈" if i == 1 else "  "
                is_match = check_result(r, test)
                match_symbol = "✅" if is_match else ""
                print(f"   {marker} {i}. {r['peripheral']}/{r['register']} (score: {r['score']:.6f}) {match_symbol}")
                
                # Show debug info if available
                debug = r.get("metadata", {}).get("_debug", {})
                if debug:
                    if use_reranker and "rerank_score" in debug:
                        print(f"       Hybrid: {debug.get('hybrid_score', 0):.6f} | Rerank: {debug.get('rerank_score', 0):.6f}")
                    else:
                        print(f"       Qdrant: {debug.get('qdrant_hybrid_score', 0):.6f}")
            
            # Show LLM answer if enabled
            if use_llm and openai_client:
                print(f"\n   🤖 LLM Answer:")
                llm_answer = query_llm_for_answer(test["query"], results, openai_client)
                print(f"   {llm_answer}")
            
            print()
    
    # Print summary by peripheral category
    print("\n" + "=" * 80)
    print("SUMMARY BY PERIPHERAL")
    print("=" * 80)
    
    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        total = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / total * 100) if total > 0 else 0
        
        status = "✅" if rate >= 80 else "⚠️" if rate >= 60 else "❌"
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
    use_llm = "--llm" in sys.argv
    
    success = run_tests(use_reranker=use_reranker, verbose=verbose, use_llm=use_llm)
    exit(0 if success else 1)
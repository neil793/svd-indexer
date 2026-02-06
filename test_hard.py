# test_realistic_queries_v2.py
"""
Realistic developer question test suite for SVD register retrieval (VERBOSE).

Goals:
- Mix "wide" inventory questions (e.g., "what timers are available?") with very specific register questions.
- Cover multiple STM32 device families / devices.
- Always show TOP 5 results for EVERY test.
- Deterministic judging (fast, CI-friendly) with optional LLM judging ("is the answer in the chunk?").

Usage:
  python test_realistic_queries_v2.py
  python test_realistic_queries_v2.py --rerank
  python test_realistic_queries_v2.py --judge-llm
  python test_realistic_queries_v2.py --llm-answer
  python test_realistic_queries_v2.py --rerank --judge-llm --llm-answer

Env:
  export OPENAI_API_KEY="..."
  export OPENAI_MODEL="gpt-4o-mini"     # optional
"""

import sys
import os
import re
from typing import List, Dict, Any, Optional
from retrieval.hybrid_retriever import HybridRetriever

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------
# TEST CASES
#
# Fields:
#   query: str
#   device_family: str (used for reporting / your own filtering if needed)
#   category: str (summary grouping)
#   peripheral: str (expected peripheral; softened for "inventory" mode)
#   tokens: List[str]  (things that must appear in chunk text to count as "correct")
#   mode: "single" | "inventory"
#   min_hits: int (inventory mode only: require >= min_hits tokens)
#   exclude_tokens: List[str] (if any appear in chunk peripheral/text, reject)
# ---------------------------------------------------------------------

TEST_CASES: List[Dict[str, Any]] = [
    # -------------------- WIDE / INVENTORY / "TIMER-LIKE" --------------------
    {
        "query": "On the STM32F030, what timers are available?",
        "device_family": "STM32F0",
        "category": "Timer",
        "peripheral": "TIM",
        "tokens": ["TIM1", "TIM3", "TIM14", "TIM16", "TIM17"],
        "mode": "inventory",
        "min_hits": 3,
    },
    {
        "query": "On the STM32F407, what GPIO ports exist?",
        "device_family": "STM32F4",
        "category": "GPIO",
        "peripheral": "GPIO",
        "tokens": ["GPIOA", "GPIOB", "GPIOC", "GPIOD", "GPIOE", "GPIOF", "GPIOG", "GPIOH", "GPIOI"],
        "mode": "inventory",
        "min_hits": 5,
    },
    {
        "query": "On the STM32F411, what DMA controllers are present?",
        "device_family": "STM32F4",
        "category": "DMA",
        "peripheral": "DMA",
        "tokens": ["DMA1", "DMA2"],
        "mode": "inventory",
        "min_hits": 1,
        "exclude_tokens": ["USB", "OTG"],
    },
    {
        "query": "On the STM32F103, what timers are available?",
        "device_family": "STM32F1",
        "category": "Timer",
        "peripheral": "TIM",
        "tokens": ["TIM1", "TIM2", "TIM3", "TIM4"],
        "mode": "inventory",
        "min_hits": 2,
    },
    {
        "query": "On the STM32F303, what ADC peripherals are available?",
        "device_family": "STM32F3",
        "category": "ADC",
        "peripheral": "ADC",
        "tokens": ["ADC1", "ADC2", "ADC3", "ADC4"],
        "mode": "inventory",
        "min_hits": 2,
    },
    {
        "query": "On the STM32F429, what USART/UART peripherals exist?",
        "device_family": "STM32F4",
        "category": "USART",
        "peripheral": "USART",
        "tokens": ["USART1", "USART2", "USART3", "UART4", "UART5", "USART6", "UART7", "UART8"],
        "mode": "inventory",
        "min_hits": 3,
    },
    {
        "query": "On the STM32F746, what SPI peripherals are available?",
        "device_family": "STM32F7",
        "category": "SPI",
        "peripheral": "SPI",
        "tokens": ["SPI1", "SPI2", "SPI3", "SPI4", "SPI5", "SPI6"],
        "mode": "inventory",
        "min_hits": 2,
    },

    # -------------------- SPECIFIC REGISTER QUESTIONS (HARDER / MORE VARIED) --------------------
    {
        "query": "Which USART register on STM32F407 is used to WRITE transmit data?",
        "device_family": "STM32F4",
        "category": "USART",
        "peripheral": "USART",
        "tokens": ["DR", "TDR", "TXDR"],
        "mode": "single",
    },
    {
        "query": "Which USART register on STM32F103 indicates TXE (transmit data register empty)?",
        "device_family": "STM32F1",
        "category": "USART",
        "peripheral": "USART",
        "tokens": ["SR", "USART_SR"],
        "mode": "single",
    },
    {
        "query": "Which USART register on STM32F103 is used to read received data?",
        "device_family": "STM32F1",
        "category": "USART",
        "peripheral": "USART",
        "tokens": ["DR", "USART_DR"],
        "mode": "single",
    },
    {
        "query": "Which SPI control register on STM32F411 enables the peripheral?",
        "device_family": "STM32F4",
        "category": "SPI",
        "peripheral": "SPI",
        "tokens": ["CR1", "SPI_CR1"],
        "mode": "single",
    },
    {
        "query": "Which SPI status register bit indicates the transmit buffer is empty on STM32F407?",
        "device_family": "STM32F4",
        "category": "SPI",
        "peripheral": "SPI",
        "tokens": ["SR", "TXE", "SPI_SR"],
        "mode": "single",
    },
    {
        "query": "Which I2C status register on STM32F429 indicates ACK or NACK?",
        "device_family": "STM32F4",
        "category": "I2C",
        "peripheral": "I2C",
        "tokens": ["SR", "ISR", "I2C_ISR", "SR1", "SR2", "NACKF", "AF"],
        "mode": "single",
    },
    {
        "query": "Which GPIO register on STM32F407 writes the output pin state atomically (set/reset)?",
        "device_family": "STM32F4",
        "category": "GPIO",
        "peripheral": "GPIO",
        "tokens": ["BSRR"],
        "mode": "single",
    },
    {
        "query": "Which GPIO register on STM32F103 configures a pin as input/output/alt-function (mode/cnf)?",
        "device_family": "STM32F1",
        "category": "GPIO",
        "peripheral": "GPIO",
        "tokens": ["CRL", "CRH", "GPIO_CRL", "GPIO_CRH"],
        "mode": "single",
    },
    {
        "query": "Which timer register on STM32F429 sets capture/compare PWM duty cycle values?",
        "device_family": "STM32F4",
        "category": "Timer",
        "peripheral": "TIM",
        "tokens": ["CCR", "CCR1", "CCR2", "CCR3", "CCR4"],
        "mode": "single",
    },
    {
        "query": "Which timer register on STM32F103 enables the counter (CEN bit)?",
        "device_family": "STM32F1",
        "category": "Timer",
        "peripheral": "TIM",
        "tokens": ["CR1", "CEN", "TIM_CR1"],
        "mode": "single",
    },
    {
        "query": "Which ADC register on STM32F407 reads the conversion result?",
        "device_family": "STM32F4",
        "category": "ADC",
        "peripheral": "ADC",
        "tokens": ["DR", "ADC_DR"],
        "mode": "single",
    },
    {
        "query": "Which ADC register on STM32F303 starts a regular conversion (software trigger)?",
        "device_family": "STM32F3",
        "category": "ADC",
        "peripheral": "ADC",
        "tokens": ["CR", "CR2", "ADSTART", "ADC_CR"],
        "mode": "single",
    },
    {
        "query": "Which DMA register on STM32F411 stores the number of data items to transfer?",
        "device_family": "STM32F4",
        "category": "DMA",
        "peripheral": "DMA",
        "tokens": ["CNDTR", "NDTR"],
        "mode": "single",
        "exclude_tokens": ["OTG", "USB"],
    },
    {
        "query": "Which DMA register holds the peripheral address for a stream on STM32F429?",
        "device_family": "STM32F4",
        "category": "DMA",
        "peripheral": "DMA",
        "tokens": ["PAR", "CPAR"],
        "mode": "single",
        "exclude_tokens": ["OTG", "USB"],
    },
    {
        "query": "Which RCC register on STM32F429 enables AHB1 peripheral clocks?",
        "device_family": "STM32F4",
        "category": "RCC",
        "peripheral": "RCC",
        "tokens": ["AHB1ENR", "ENR"],
        "mode": "single",
    },
    {
        "query": "Which RCC register on STM32F103 enables APB2 peripheral clocks?",
        "device_family": "STM32F1",
        "category": "RCC",
        "peripheral": "RCC",
        "tokens": ["APB2ENR", "RCC_APB2ENR"],
        "mode": "single",
    },
    {
        "query": "Which EXTI register enables an interrupt line on STM32F407?",
        "device_family": "STM32F4",
        "category": "EXTI",
        "peripheral": "EXTI",
        "tokens": ["IMR", "EXTI_IMR"],
        "mode": "single",
    },
    {
        "query": "Which EXTI register selects rising edge trigger on STM32F103?",
        "device_family": "STM32F1",
        "category": "EXTI",
        "peripheral": "EXTI",
        "tokens": ["RTSR", "EXTI_RTSR"],
        "mode": "single",
    },
    {
        "query": "Which RTC register on STM32F429 stores the current TIME?",
        "device_family": "STM32F4",
        "category": "RTC",
        "peripheral": "RTC",
        "tokens": ["TR", "RTC_TR", "TSTR"],
        "mode": "single",
    },
    {
        "query": "Which IWDG register on STM32F407 reloads the watchdog counter?",
        "device_family": "STM32F4",
        "category": "IWDG",
        "peripheral": "IWDG",
        "tokens": ["KR", "IWDG_KR"],
        "mode": "single",
    },

    # -------------------- WIDE / CLOCK / POWER / NVIC-LIKE QUESTIONS --------------------
    # These are "wide" but still judged by token presence in chunk text.
    {
        "query": "On the STM32F429, which RCC registers control the system clock configuration (PLL/clock switch)?",
        "device_family": "STM32F4",
        "category": "RCC",
        "peripheral": "RCC",
        "tokens": ["CFGR", "PLLCFGR", "RCC_CFGR", "RCC_PLLCFGR"],
        "mode": "inventory",
        "min_hits": 2,
    },
    {
        "query": "On the STM32F103, which RCC registers configure PLL and clock switching?",
        "device_family": "STM32F1",
        "category": "RCC",
        "peripheral": "RCC",
        "tokens": ["CFGR", "CR", "RCC_CFGR", "RCC_CR", "PLLSRC", "PLL"],
        "mode": "inventory",
        "min_hits": 2,
    },
    {
        "query": "On the STM32F407, what NVIC-related registers exist for enabling and prioritizing interrupts?",
        "device_family": "STM32F4",
        "category": "NVIC",
        "peripheral": "NVIC",
        "tokens": ["ISER", "ICER", "IPR", "NVIC_ISER", "NVIC_IPR"],
        "mode": "inventory",
        "min_hits": 2,
    },
]


# ---------------------------------------------------------------------
# JUDGING HELPERS
# ---------------------------------------------------------------------

def _norm(s: Optional[str]) -> str:
    return (s or "").upper()

def _word_match(token: str, text: str) -> bool:
    """
    Token match with boundaries.
    Boundary definition: start/end or any non [A-Z0-9_]
    This avoids substring traps like DR inside "ADDR".
    """
    token_u = re.escape(_norm(token))
    text_u = _norm(text)
    pattern = rf"(?<![A-Z0-9_]){token_u}(?![A-Z0-9_])"
    return re.search(pattern, text_u) is not None

def _count_hits(tokens: List[str], text: str) -> int:
    return sum(1 for t in tokens if _word_match(t, text))

def check_result_deterministic(result: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """
    Deterministic judge:
      - mode=single: pass if >= 1 token appears in chunk text, and peripheral matches (softened)
      - mode=inventory: pass if >= min_hits tokens appear in chunk text (peripheral check softened)
    """
    mode = test.get("mode", "single")
    min_hits = int(test.get("min_hits", 1))

    peripheral_expected = _norm(test.get("peripheral"))
    peripheral_got = _norm(result.get("peripheral"))
    text = _norm(result.get("text"))

    # Exclusion tokens apply to BOTH peripheral field and text.
    for ex in test.get("exclude_tokens", []):
        ex_u = _norm(ex)
        if ex_u and (ex_u in peripheral_got or ex_u in text):
            return False

    # Peripheral gating:
    # - For single-register questions, require expected peripheral appears either in peripheral field OR in text.
    # - For inventory questions, don't hard-fail if peripheral field is weird (device_summary etc).
    if peripheral_expected:
        if mode == "single":
            if peripheral_expected not in peripheral_got and peripheral_expected not in text:
                return False
        else:
            # inventory: allow passing purely by token coverage even if peripheral label is nonstandard
            pass

    hits = _count_hits(test["tokens"], text)

    if mode == "inventory":
        return hits >= min_hits
    return hits >= 1


def llm_judge_contains_answer(query: str, result: Dict[str, Any], openai_client) -> bool:
    """
    LLM judge: does THIS chunk contain enough explicit info to answer the question?
    Returns True/False based on YES/NO.
    """
    peripheral = result.get("peripheral", "Unknown")
    chunk_type = result.get("type", "unknown")
    text = (result.get("text") or "")[:1600]  # give judge more room

    system_prompt = (
        "You are grading retrieval results for an embedded systems Q&A system.\n"
        "Answer ONLY 'YES' or 'NO'.\n"
        "Say YES only if the chunk contains enough explicit information to answer the question.\n"
        "Do not guess or use outside knowledge."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Chunk metadata: peripheral={peripheral}, type={chunk_type}\n"
        f"Chunk text:\n{text}\n\n"
        "Does this chunk contain the answer?"
    )

    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("YES")
    except Exception:
        return False


def query_llm_for_answer(query: str, top_results: List[Dict[str, Any]], openai_client) -> str:
    """
    Use OpenAI to answer the user's question based on retrieval results.
    """
    context_parts = []
    for i, result in enumerate(top_results[:5], 1):
        peripheral = result.get("peripheral", "Unknown")
        chunk_type = result.get("type", "unknown")
        text = (result.get("text", "No text available") or "")[:700]
        context_parts.append(
            f"Result {i}: {peripheral} (type={chunk_type})\n"
            f"Content:\n{text}\n"
        )

    context = "\n".join(context_parts)

    system_prompt = (
        "You are an expert embedded systems engineer helping with STM32 microcontroller register questions.\n"
        "Given search results from an SVD database, provide a clear, concise answer.\n"
        "If it is an inventory question, list the peripherals found in the context.\n"
        "Do not invent peripherals or registers that are not present in the provided context."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Search Results:\n{context}\n\n"
        "Answer using ONLY what is present above."
    )

    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying LLM: {str(e)}"


# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------

def run_tests(use_reranker: bool = False, use_llm_answer: bool = False, use_llm_judge: bool = False) -> bool:
    print("\n" + "=" * 80)
    print("SVD REGISTER RETRIEVAL TEST SUITE - VERBOSE MODE (v2)")
    print("Shows top 5 results for EVERY test")
    if use_reranker:
        print("🔥 RERANKING ENABLED")
    if use_llm_judge:
        print("🧑‍⚖️  LLM JUDGE ENABLED (YES/NO: answer in chunk?)")
    if use_llm_answer:
        print("🤖 LLM ANSWER GENERATION ENABLED")
    print("=" * 80)

    # Initialize OpenAI if requested
    openai_client = None
    if use_llm_answer or use_llm_judge:
        if not OPENAI_AVAILABLE:
            print("⚠️  OpenAI not installed. Install with: pip install openai")
            print("    Proceeding WITHOUT LLM features.\n")
            use_llm_answer = False
            use_llm_judge = False
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not set. Proceeding WITHOUT LLM features.\n")
                use_llm_answer = False
                use_llm_judge = False
            else:
                openai_client = OpenAI(api_key=api_key)
                print("✓ OpenAI initialized\n")

    retriever = HybridRetriever(use_reranker=use_reranker)

    results_by_category: Dict[str, Dict[str, int]] = {}
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
            print(f"\n{'=' * 80}")
            print(f"❌ NO RESULTS: {test['query']}")
            print(f"📍 Expected peripheral: {test.get('peripheral', 'N/A')} | mode={test.get('mode','single')}")
            print(f"📍 Tokens: {test.get('tokens', [])}")
            print(f"{'=' * 80}")
            continue

        top_result = results[0]

        # Top-1 correctness (deterministic, with optional LLM judge fallback)
        is_correct = check_result_deterministic(top_result, test)
        if (not is_correct) and use_llm_judge and openai_client:
            is_correct = llm_judge_contains_answer(test["query"], top_result, openai_client)

        # Find first correct rank in top-5
        correct_rank = None
        for i, r in enumerate(results[:5], 1):
            ok = check_result_deterministic(r, test)
            if (not ok) and use_llm_judge and openai_client:
                ok = llm_judge_contains_answer(test["query"], r, openai_client)
            if ok:
                correct_rank = i
                break

        if is_correct:
            passed += 1
            results_by_category[category]["passed"] += 1
            status = "✅ PASS"
        else:
            failed += 1
            results_by_category[category]["failed"] += 1
            status = "❌ FAIL"

        rank_info = f"(correct at #{correct_rank})" if correct_rank else "(not in top 5)"

        print(f"\n{'=' * 80}")
        print(f"{status}: {test['query']}")
        print(f"📍 Device family: {test.get('device_family','N/A')} | Category: {category}")
        print(f"📍 Expected peripheral: {test.get('peripheral','N/A')} | mode={test.get('mode','single')}")
        if test.get("mode") == "inventory":
            print(f"📍 Inventory tokens (min_hits={test.get('min_hits', 1)}): {test.get('tokens', [])}")
        else:
            print(f"📍 Acceptable tokens: {test.get('tokens', [])}")
        print(f"{'=' * 80}")
        print(f"   Top result: {top_result.get('peripheral', 'None')} (type: {top_result.get('type', 'unknown')})")
        print(f"   Score: {top_result.get('score', 0.0):.6f}")
        if not is_correct:
            print(f"   {rank_info}")

        print(f"\n   Top 5 results:")
        for i, r in enumerate(results[:5], 1):
            marker = "👈" if i == 1 else "  "
            det_ok = check_result_deterministic(r, test)
            llm_ok = None

            if (not det_ok) and use_llm_judge and openai_client:
                llm_ok = llm_judge_contains_answer(test["query"], r, openai_client)

            final_ok = det_ok or (llm_ok is True)

            match_symbol = "✅" if final_ok else ""
            peripheral = r.get("peripheral", "None")
            chunk_type = r.get("type", "unknown")
            score = r.get("score", 0.0)

            print(f"   {marker} {i}. {peripheral} [{chunk_type}] (score: {score:.6f}) {match_symbol}")

            debug = r.get("metadata", {}).get("_debug", {})
            if debug:
                if use_reranker and "rerank_score" in debug:
                    hybrid = debug.get("hybrid_score", 0)
                    rerank = debug.get("rerank_score", 0)
                    print(f"       Hybrid: {hybrid:.6f} | Rerank: {rerank:.6f}")
                else:
                    qdrant = debug.get("qdrant_hybrid_score", 0)
                    print(f"       Qdrant: {qdrant:.6f}")

                applied = debug.get("applied_boosts", {})
                if applied.get("peripheral_match"):
                    print("       ✓ Peripheral boost applied")
                if applied.get("register_match"):
                    print("       ✓ Token/register boost applied")
                if applied.get("peripheral_penalty"):
                    print("       ⚠️  Peripheral penalty applied")

        if use_llm_answer and openai_client:
            print(f"\n   🤖 LLM Answer:")
            llm_answer = query_llm_for_answer(test["query"], results, openai_client)
            print(f"   {llm_answer}")

        print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)

    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        total = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / total * 100) if total > 0 else 0.0
        status = "✅" if rate >= 80 else "⚠️" if rate >= 60 else "❌"
        print(f"{status} {category:12} {stats['passed']:2}/{total:2} ({rate:5.1f}%)")

    print("\n" + "=" * 80)
    print("OVERALL")
    print("=" * 80)
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0.0
    print(f"Total: {total} | ✅ {passed} | ❌ {failed} | Rate: {success_rate:.1f}%")

    if use_reranker:
        print("\n💡 Reranker was used to rescore candidates.")
    if use_llm_judge:
        print("💡 LLM judge was used as a fallback when deterministic judge said NO.")
    if use_llm_answer:
        print("💡 LLM answer generation was enabled.")

    print()
    return success_rate >= 70.0


if __name__ == "__main__":
    use_reranker = "--rerank" in sys.argv
    use_llm_judge = "--judge-llm" in sys.argv
    use_llm_answer = "--llm-answer" in sys.argv

    ok = run_tests(use_reranker=use_reranker, use_llm_answer=use_llm_answer, use_llm_judge=use_llm_judge)
    raise SystemExit(0 if ok else 1)

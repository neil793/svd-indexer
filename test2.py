# test_realistic_queries.py

"""
Realistic developer question test suite for SVD register retrieval.
Tests common embedded development queries against the hybrid search system.
"""

import sys
from typing import List, Dict, Any, Set
from retrieval.hybrid_retriever import HybridRetriever


# Test cases with multiple acceptable register name variants
TEST_CASES = [
    # USART/UART
    {
        "query": "Which USART register is used to WRITE transmit data?",
        "peripheral": "USART",
        "registers": ["DR", "TDR", "TXDR"],  # Accept any variant
        "category": "USART"
    },
    {
        "query": "Which USART register is used to READ received data?",
        "peripheral": "USART",
        "registers": ["DR", "RDR", "RXDR"],
        "category": "USART"
    },
    {
        "query": "Which USART STATUS register bit indicates transmission complete?",
        "peripheral": "USART",
        "registers": ["SR", "ISR", "ISR_FIFO_DISABLED", "ISR_FIFO_ENABLED"],
        "category": "USART"
    },
    {
        "query": "Which USART STATUS register bit indicates receive data ready?",
        "peripheral": "USART",
        "registers": ["SR", "ISR", "ISR_FIFO_DISABLED", "ISR_FIFO_ENABLED"],
        "category": "USART"
    },
    {
        "query": "Which USART CONTROL register enables the peripheral?",
        "peripheral": "USART",
        "registers": ["CR1", "USART_CR1", "USART_CR1_ALTERNATE"],
        "category": "USART"
    },
    
    # SPI
    {
        "query": "Which SPI register is used to WRITE transmit data?",
        "peripheral": "SPI",
        "registers": ["DR", "TXDR", "SPI_DR"],
        "category": "SPI"
    },
    {
        "query": "Which SPI register is used to READ received data?",
        "peripheral": "SPI",
        "registers": ["DR", "RXDR", "SPI_DR"],
        "category": "SPI"
    },
    {
        "query": "Which SPI STATUS register indicates the peripheral is busy?",
        "peripheral": "SPI",
        "registers": ["SR", "SPI_SR"],
        "category": "SPI"
    },
    {
        "query": "Which SPI CONTROL register enables the peripheral?",
        "peripheral": "SPI",
        "registers": ["CR1", "SPI_CR1"],
        "category": "SPI"
    },
    
    # I2C
    {
        "query": "Which I2C register is used to WRITE transmit data?",
        "peripheral": "I2C",
        "registers": ["DR", "TXDR", "I2C_TXDR"],
        "category": "I2C"
    },
    {
        "query": "Which I2C STATUS register indicates ACK or NACK?",
        "peripheral": "I2C",
        "registers": ["SR", "ISR", "I2C_ISR", "SR1", "SR2"],
        "category": "I2C"
    },
    {
        "query": "Which I2C register stores the slave address?",
        "peripheral": "I2C",
        "registers": ["OAR", "OAR1", "OAR2", "I2C_OAR1", "I2C_OAR2"],
        "category": "I2C"
    },
    
    # GPIO
    {
        "query": "Which GPIO register READS the input pin state?",
        "peripheral": "GPIO",
        "registers": ["IDR"],
        "category": "GPIO"
    },
    {
        "query": "Which GPIO register WRITES the output pin state?",
        "peripheral": "GPIO",
        "registers": ["ODR"],
        "category": "GPIO"
    },
    {
        "query": "Which GPIO register configures pin mode as input or output?",
        "peripheral": "GPIO",
        "registers": ["MODER"],
        "category": "GPIO"
    },
    {
        "query": "Which GPIO register configures pull-up or pull-down resistors?",
        "peripheral": "GPIO",
        "registers": ["PUPDR"],
        "category": "GPIO"
    },
    {
        "query": "Which GPIO register configures alternate pin functions?",
        "peripheral": "GPIO",
        "registers": ["AFR", "AFRL", "AFRH"],
        "category": "GPIO"
    },
    
    # Timer
    {
        "query": "Which timer register READS the current counter value?",
        "peripheral": "TIM",
        "registers": ["CNT"],
        "exclude_peripherals": ["STK"],  # Exclude SysTick
        "category": "Timer"
    },
    {
        "query": "Which timer register sets the auto-reload (period) value?",
        "peripheral": "TIM",
        "registers": ["ARR"],
        "category": "Timer"
    },
    {
        "query": "Which timer register sets the prescaler value?",
        "peripheral": "TIM",
        "registers": ["PSC"],
        "exclude_peripherals": ["WWDG", "IWDG"],  # Exclude watchdogs
        "category": "Timer"
    },
    {
        "query": "Which timer register sets capture/compare (PWM duty cycle) values?",
        "peripheral": "TIM",
        "registers": ["CCR", "CCR1", "CCR2", "CCR3", "CCR4"],
        "category": "Timer"
    },
    {
        "query": "Which timer STATUS register indicates an update or overflow event?",
        "peripheral": "TIM",
        "registers": ["SR", "ISR", "ISR_output"],
        "category": "Timer"
    },
    
    # ADC
    {
        "query": "Which ADC register READS the conversion result?",
        "peripheral": "ADC",
        "registers": ["DR", "ADC_DR"],
        "category": "ADC"
    },
    {
        "query": "Which ADC CONTROL register starts a conversion?",
        "peripheral": "ADC",
        "registers": ["CR", "CR1", "CR2"],
        "category": "ADC"
    },
    {
        "query": "Which ADC register configures the sampling time?",
        "peripheral": "ADC",
        "registers": ["SMPR", "SMPR1", "SMPR2"],
        "category": "ADC"
    },
    {
        "query": "Which ADC STATUS register indicates conversion complete?",
        "peripheral": "ADC",
        "registers": ["SR", "ISR", "ADC_ISR"],
        "category": "ADC"
    },
    
    # DMA
    {
        "query": "Which DMA register stores the number of data items to transfer?",
        "peripheral": "DMA",
        "registers": ["CNDTR", "NDTR"],
        "exclude_peripherals": ["OTG", "USB"],  # Exclude USB DMA
        "category": "DMA"
    },
    {
        "query": "Which DMA register stores the peripheral source address?",
        "peripheral": "DMA",
        "registers": ["CPAR", "PAR"],
        "exclude_peripherals": ["OTG", "USB"],
        "category": "DMA"
    },
    {
        "query": "Which DMA register stores the memory destination address?",
        "peripheral": "DMA",
        "registers": ["CMAR", "MAR"],
        "exclude_peripherals": ["OTG", "USB"],
        "category": "DMA"
    },
    {
        "query": "Which DMA CONTROL register enables a DMA channel?",
        "peripheral": "DMA",
        "registers": ["CCR"],
        "exclude_peripherals": ["OTG", "USB"],
        "category": "DMA"
    },
    
    # RCC
    {
        "query": "Which RCC register ENABLES peripheral clocks?",
        "peripheral": "RCC",
        "registers": ["ENR", "AHB1ENR", "AHB2ENR", "APB1ENR", "APB2ENR", "APB4ENR"],
        "category": "RCC"
    },
    {
        "query": "Which RCC register configures the system clock source and prescalers?",
        "peripheral": "RCC",
        "registers": ["CFGR"],
        "category": "RCC"
    },
    {
        "query": "Which RCC STATUS register indicates PLL ready?",
        "peripheral": "RCC",
        "registers": ["CR", "RCC_CR"],
        "category": "RCC"
    },
    
    # RTC
    {
        "query": "Which RTC register stores the current TIME?",
        "peripheral": "RTC",
        "registers": ["TR", "RTC_TR", "TSTR"],
        "category": "RTC"
    },
    {
        "query": "Which RTC register stores the current DATE?",
        "peripheral": "RTC",
        "registers": ["DR", "RTC_DR", "TSDR"],
        "category": "RTC"
    },
    {
        "query": "Which RTC register configures alarms?",
        "peripheral": "RTC",
        "registers": ["ALRM", "ALRMAR", "ALRMBR", "ALRABINR", "ALRBBINR"],
        "category": "RTC"
    },
    
    # Flash
    {
        "query": "Which FLASH CONTROL register starts erase or program operations?",
        "peripheral": "Flash",
        "registers": ["CR", "FLASH_CR"],
        "category": "Flash"
    },
    {
        "query": "Which FLASH STATUS register indicates operation complete?",
        "peripheral": "Flash",
        "registers": ["SR", "FLASH_SR", "OPSR", "FLASH_OPSR"],
        "category": "Flash"
    },
    
    # PWR
    {
        "query": "Which PWR CONTROL register enters low-power modes?",
        "peripheral": "PWR",
        "registers": ["CR", "CR1", "PWR_CR1"],
        "category": "PWR"
    },
    {
        "query": "Which PWR CONTROL register enables backup domain access?",
        "peripheral": "PWR",
        "registers": ["CR", "CR1", "PWR_CR1", "DBPR", "PWR_DBPR"],
        "category": "PWR"
    },
    
    # IWDG
    {
        "query": "Which IWDG register reloads the watchdog counter?",
        "peripheral": "IWDG",
        "registers": ["KR", "IWDG_KR"],
        "category": "IWDG"
    },
    {
        "query": "Which IWDG register sets the watchdog timeout value?",
        "peripheral": "IWDG",
        "registers": ["RLR", "IWDG_RLR"],
        "category": "IWDG"
    },
    
    # CRC
    {
        "query": "Which CRC register READS or WRITES the CRC data value?",
        "peripheral": "CRC",
        "registers": ["DR", "CRC_DR"],
        "category": "CRC"
    },
    {
        "query": "Which CRC CONTROL register resets the CRC calculation unit?",
        "peripheral": "CRC",
        "registers": ["CR", "CRC_CR"],
        "category": "CRC"
    },
]


def check_result(result: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """
    Check if result matches test expectations.
    
    Args:
        result: Top search result with peripheral and register fields
        test: Test case with expected values
    
    Returns:
        True if result matches expectations
    """
    peripheral = (result.get("peripheral") or "").upper()
    register = (result.get("register") or "").upper()
    
    # Check peripheral match (must contain expected peripheral)
    expected_periph = test["peripheral"].upper()
    if expected_periph not in peripheral:
        return False
    
    # Check if peripheral should be excluded
    if "exclude_peripherals" in test:
        for excluded in test["exclude_peripherals"]:
            if excluded.upper() in peripheral:
                return False
    
    # Check register match (any variant is acceptable)
    for acceptable_reg in test["registers"]:
        if acceptable_reg.upper() in register:
            return True
    
    return False


def run_tests(use_reranker: bool = False):
    """Run all test cases and report results."""
    print("\n" + "=" * 80)
    print("SVD REGISTER RETRIEVAL TEST SUITE")
    if use_reranker:
        print("🔥 RERANKING ENABLED (Cross-Encoder)")
    print("=" * 80)
    
    retriever = HybridRetriever(use_reranker=use_reranker)
    
    results_by_category = {}
    passed = 0
    failed = 0
    failed_tests = []
    
    for test in TEST_CASES:
        category = test["category"]
        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "failed": 0, "tests": []}
        
        print(f"\n{'=' * 80}")
        print(f"❓ {test['query']}")
        print(f"📍 Expected: {test['peripheral']}/{'/'.join(test['registers'])}")
        
        # Exclude info if present
        if "exclude_peripherals" in test:
            print(f"🚫 Exclude: {', '.join(test['exclude_peripherals'])}")
        
        print("=" * 80)
        
        # Perform search
        results = retriever.search(test["query"], top_k=5)
        
        if not results:
            print("❌ No results returned")
            failed += 1
            results_by_category[category]["failed"] += 1
            failed_tests.append({
                "test": test,
                "reason": "No results",
                "top_5": []
            })
            continue
        
        top_result = results[0]
        is_correct = check_result(top_result, test)
        
        if is_correct:
            print(f"✅ PASS: {top_result['peripheral']}/{top_result['register']}")
            print(f"   Score: {top_result['score']:.6f}")
            passed += 1
            results_by_category[category]["passed"] += 1
            results_by_category[category]["tests"].append({
                "query": test["query"],
                "passed": True
            })
        else:
            print(f"❌ FAIL: Got {top_result['peripheral']}/{top_result['register']}")
            print(f"   Score: {top_result['score']:.6f}")
            print(f"\n   Top 5 results:")
            
            for i, r in enumerate(results[:5], 1):
                marker = "👈" if i == 1 else "  "
                is_match = check_result(r, test)
                match_symbol = "✅" if is_match else ""
                print(f"   {marker} {i}. {r['peripheral']}/{r['register']} (score: {r['score']:.6f}) {match_symbol}")
                
                # Show debug info
                debug = r.get("metadata", {}).get("_debug", {})
                if debug:
                    if use_reranker and "rerank_score" in debug:
                        print(f"       Hybrid: {debug.get('hybrid_score', 0):.6f} | Rerank: {debug.get('rerank_score', 0):.6f}")
                    else:
                        print(f"       Qdrant: {debug.get('qdrant_hybrid_score', 0):.6f}")
            
            failed += 1
            results_by_category[category]["failed"] += 1
            results_by_category[category]["tests"].append({
                "query": test["query"],
                "passed": False
            })
            failed_tests.append({
                "test": test,
                "got": f"{top_result['peripheral']}/{top_result['register']}",
                "top_5": [(r['peripheral'], r['register'], r['score']) for r in results[:5]]
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    
    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        total = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / total * 100) if total > 0 else 0
        print(f"\n{category}:")
        print(f"  ✅ Passed: {stats['passed']}/{total} ({rate:.1f}%)")
        print(f"  ❌ Failed: {stats['failed']}/{total}")
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if use_reranker:
        print("\n💡 Reranker was used to rescore top results")
    
    # Show worst performing categories
    if failed > 0:
        print("\n" + "=" * 80)
        print("CATEGORIES NEEDING IMPROVEMENT")
        print("=" * 80)
        
        category_rates = []
        for category, stats in results_by_category.items():
            total = stats["passed"] + stats["failed"]
            rate = (stats["passed"] / total * 100) if total > 0 else 0
            category_rates.append((category, rate, stats["failed"]))
        
        category_rates.sort(key=lambda x: x[1])  # Sort by success rate
        
        for category, rate, num_failed in category_rates[:5]:
            if rate < 100:
                print(f"  {category}: {rate:.1f}% ({num_failed} failures)")
    
    print("\n")
    return success_rate >= 70.0  # Return True if 70%+ pass rate


if __name__ == "__main__":
    # Check if --rerank flag is passed
    use_reranker = "--rerank" in sys.argv
    
    success = run_tests(use_reranker=use_reranker)
    exit(0 if success else 1)
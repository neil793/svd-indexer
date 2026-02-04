"""
Create searchable text chunks from parsed SVD registers
DUAL-LEVEL CHUNKING: Optimized for 512-token embedding models like all-MiniLM-L6-v2

Strategy:
- Summary chunks: ~300-400 chars (all register names, configuration hints)
- Detail chunks: ~400 chars each (3-8 registers with full field details)
"""
from typing import List, Dict, Any
from collections import defaultdict
import hashlib
from .models import ParsedRegister, TextChunk
from .config import config

# Token limits for all-MiniLM-L6-v2 (512 tokens max)
MAX_SUMMARY_CHARS = 400  # ~512 tokens
MAX_DETAIL_CHARS = 400   # ~512 tokens per detail chunk


def _group_registers_by_peripheral(registers: List[ParsedRegister]) -> Dict[str, List[ParsedRegister]]:
    """
    Group registers by their peripheral.
    
    Returns:
        Dict mapping peripheral_key -> list of registers
        Key format: "device/peripheral" or "peripheral" (if deduplicated)
    """
    groups = defaultdict(list)
    
    for reg in registers:
        if hasattr(reg, 'devices') and reg.devices and len(reg.devices) > 1:
            # Deduplicated - use peripheral name only
            key = reg.peripheral
        else:
            # Device-specific
            key = f"{reg.device}/{reg.peripheral}"
        
        groups[key].append(reg)
    
    return groups


def _categorize_registers(registers: List[ParsedRegister]) -> Dict[str, List[ParsedRegister]]:
    """
    Categorize registers by their likely function.
    
    Categories:
    - Control: CR, CTL, CTRL registers
    - Status: SR, STAT, STATUS registers  
    - Data: DR, DATA, TX, RX registers
    - Configuration: CFG, CONFIG registers
    - Other: Everything else
    """
    categories = {
        "control": [],
        "status": [],
        "data": [],
        "configuration": [],
        "other": []
    }
    
    for reg in registers:
        reg_upper = reg.register.upper()
        
        if any(x in reg_upper for x in ["CR", "CTL", "CTRL"]):
            categories["control"].append(reg)
        elif any(x in reg_upper for x in ["SR", "STAT", "STATUS", "ISR", "FLAG"]):
            categories["status"].append(reg)
        elif any(x in reg_upper for x in ["DR", "DATA", "TX", "RX", "BUF"]):
            categories["data"].append(reg)
        elif any(x in reg_upper for x in ["CFG", "CONFIG", "CONF"]):
            categories["configuration"].append(reg)
        else:
            categories["other"].append(reg)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def _format_register_detailed(register: ParsedRegister) -> str:
    """Format a single register with full field details."""
    lines = []
    
    # Register header
    lines.append(f"\n{register.register}")
    if register.register_description:
        lines.append(f"  {register.register_description}")
    
    lines.append(f"  Address: {register.full_address}")
    
    if register.access:
        lines.append(f"  Access: {register.access}")
    if register.reset_value:
        lines.append(f"  Reset: {register.reset_value}")
    
    # Fields (condensed)
    if config.include_field_names and register.fields:
        field_strs = []
        for field in register.fields:
            field_str = f"{field.name}{field.bit_range}"
            if config.include_field_desc and field.description:
                field_str += f" - {field.description}"
            field_strs.append(field_str)
        lines.append(f"  Fields: {', '.join(field_strs)}")
    
    return "\n".join(lines)


def _get_configuration_hints(peripheral: str) -> str:
    """Get configuration hints for a peripheral type"""
    peripheral_upper = peripheral.upper()
    
    if "USART" in peripheral_upper or "UART" in peripheral_upper:
        return ("Enable clock (RCC) → Configure baud rate (BRR) → "
                "Enable TX/RX (CR1) → Enable UART (CR1.UE). "
                "Key: CR1, DR, SR, BRR")
    
    elif "SPI" in peripheral_upper:
        return ("Enable clock (RCC) → Configure mode/clock (CR1) → "
                "Enable SPI (CR1.SPE) → Write to DR. "
                "Key: CR1, CR2, DR, SR")
    
    elif "I2C" in peripheral_upper:
        return ("Enable clock (RCC) → Configure timing (TIMINGR/CCR) → "
                "Enable I2C (CR1.PE) → Generate START. "
                "Key: CR1, CR2, SR1, SR2")
    
    elif "TIM" in peripheral_upper:
        return ("Enable clock (RCC) → Set prescaler (PSC) → "
                "Set auto-reload (ARR) → Enable counter (CR1.CEN). "
                "PWM: Configure CCR → Set mode (CCMR) → Enable (CCER)")
    
    elif "GPIO" in peripheral_upper:
        return ("Configure mode (MODER) → Set output (BSRR) → "
                "Read input (IDR) → Pull-up/down (PUPDR)")
    
    elif "ADC" in peripheral_upper:
        return ("Enable clock (RCC) → Configure channels (SQR) → "
                "Start conversion (CR2.SWSTART) → Read result (DR)")
    
    elif "DMA" in peripheral_upper:
        return ("Configure addresses (PAR/MAR) → Set count (NDTR) → "
                "Configure mode (CCR) → Enable (CCR.EN)")
    
    elif "RCC" in peripheral_upper:
        return "System clock configuration and peripheral clock enables"
    
    elif "PWR" in peripheral_upper:
        return "Power control, low-power modes, voltage regulation"
    
    return f"{peripheral}: System peripheral"


def create_peripheral_summary_chunk(
    peripheral_key: str,
    registers: List[ParsedRegister]
) -> TextChunk:
    """
    Create a summary chunk for a peripheral (~300-400 chars, fits in 512 tokens).
    
    Optimized for broad queries like:
    - "What registers does USART have?"
    - "How do I configure SPI?"
    - "What peripherals control timers?"
    
    Args:
        peripheral_key: Key identifying the peripheral
        registers: All registers for this peripheral
        
    Returns:
        Summary TextChunk optimized for 512-token models
    """
    rep = registers[0]
    lines = []
    
    # CRITICAL INFO FIRST (most important for search)
    lines.append(f"Peripheral: {rep.peripheral}")
    if config.include_peripheral_desc and rep.peripheral_description:
        lines.append(f"Description: {rep.peripheral_description[:100]}")  # Truncate desc
    
    if config.include_device:
        lines.append(f"Device: {rep.device}")
    
    # All register names (MOST CRITICAL - always include)
    reg_names = sorted([r.register for r in registers])
    lines.append(f"\nRegisters ({len(registers)}): {', '.join(reg_names)}")
    
    # Check if we're approaching limit
    current_text = "\n".join(lines)
    remaining_chars = MAX_SUMMARY_CHARS - len(current_text)
    
    # Only add more if we have space
    if remaining_chars > 100:
        # Categorized register list (condensed)
        categories = _categorize_registers(registers)
        for cat_name, cat_regs in categories.items():
            cat_reg_names = sorted([r.register for r in cat_regs])
            cat_line = f"{cat_name.title()}: {', '.join(cat_reg_names)}"
            
            # Check if adding this would exceed limit
            if len(current_text) + len(cat_line) + 1 < MAX_SUMMARY_CHARS - 50:  # Leave 50 char buffer
                lines.append(cat_line)
                current_text = "\n".join(lines)
            else:
                break  # Stop adding categories
    
    # Check remaining space for fields
    remaining_chars = MAX_SUMMARY_CHARS - len(current_text)
    if remaining_chars > 80:
        # Add some field names
        all_fields = set()
        for reg in registers:
            all_fields.update([f.name for f in reg.fields])
        
        if all_fields:
            # Only include as many fields as fit
            field_list = sorted(all_fields)
            field_str = ', '.join(field_list)
            
            # Truncate field list if too long
            max_field_len = remaining_chars - 20  # "Common fields: " + buffer
            if len(field_str) > max_field_len:
                # Truncate and add ellipsis
                field_str = field_str[:max_field_len-3] + "..."
            
            lines.append(f"\nCommon fields: {field_str}")
            current_text = "\n".join(lines)
    
    # Configuration hints (only if space remains)
    remaining_chars = MAX_SUMMARY_CHARS - len(current_text)
    if remaining_chars > 60:
        hints = _get_configuration_hints(rep.peripheral)
        # Truncate hints if needed
        if len(hints) > remaining_chars - 10:
            hints = hints[:remaining_chars-13] + "..."
        lines.append(f"\nConfig: {hints}")
    
    # Final text with hard limit enforcement
    text = "\n".join(lines)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS-3] + "..."
    
    # Metadata (unchanged)
    all_fields = set()
    for reg in registers:
        all_fields.update([f.name for f in reg.fields])
    
    metadata: Dict[str, Any] = {
        "type": "peripheral_summary",
        "device": rep.device,
        "peripheral": rep.peripheral,
        "peripheral_group": rep.peripheral_group or "",
        "register_count": len(registers),
        "registers": [r.register for r in registers],
        "field_names": ", ".join(sorted(all_fields)),
        "peripheral_lower": (rep.peripheral or "").lower(),
        "devices": getattr(rep, 'devices', [rep.device]),
        "peripheral_instances": getattr(rep, 'peripheral_instances', [rep.peripheral]),
    }
    
    chunk_id = f"{rep.peripheral}_summary_{hashlib.md5(peripheral_key.encode()).hexdigest()[:8]}"
    
    return TextChunk(id=chunk_id, text=text, metadata=metadata)


def _create_detail_chunk(
    peripheral_key: str,
    rep: ParsedRegister,
    registers: List[ParsedRegister],
    chunk_index: int
) -> TextChunk:
    """Helper to create a single detail chunk with size enforcement"""
    lines = []
    
    # Minimal header (keep tokens for content)
    lines.append(f"Peripheral: {rep.peripheral} (Detail {chunk_index + 1})")
    lines.append(f"Device: {rep.device}")
    
    # Register list for this chunk
    reg_names = [r.register for r in registers]
    lines.append(f"Registers: {', '.join(reg_names)}\n")
    
    # Add registers one by one, checking size
    current_text = "\n".join(lines)
    
    for reg in registers:
        reg_text = _format_register_detailed(reg)
        
        # Check if adding this register would exceed limit
        if len(current_text) + len(reg_text) + 1 < MAX_DETAIL_CHARS:
            lines.append(reg_text)
            current_text = "\n".join(lines)
        else:
            # This register would exceed limit, stop here
            lines.append("\n[Additional registers truncated to fit 512 token limit]")
            break
    
    # Final text with hard limit
    text = "\n".join(lines)
    if len(text) > MAX_DETAIL_CHARS:
        text = text[:MAX_DETAIL_CHARS-3] + "..."
    
    # Collect field names for this chunk
    all_fields = set()
    for reg in registers:
        all_fields.update([f.name for f in reg.fields])
    
    metadata: Dict[str, Any] = {
        "type": "peripheral_detail",
        "device": rep.device,
        "peripheral": rep.peripheral,
        "peripheral_group": rep.peripheral_group or "",
        "chunk_part": chunk_index + 1,
        "register_count": len(registers),
        "registers": [r.register for r in registers],
        "field_names": ", ".join(sorted(all_fields)),
        "peripheral_lower": (rep.peripheral or "").lower(),
        "devices": getattr(rep, 'devices', [rep.device]),
        "peripheral_instances": getattr(rep, 'peripheral_instances', [rep.peripheral]),
    }
    
    chunk_id = f"{rep.peripheral}_detail_{chunk_index}_{hashlib.md5(peripheral_key.encode()).hexdigest()[:8]}"
    
    return TextChunk(id=chunk_id, text=text, metadata=metadata)


def create_device_summary_chunks(registers: List[ParsedRegister]) -> List[TextChunk]:
    # Group peripherals by device
    device_peripherals = defaultdict(set)
    device_series = {}

    for reg in registers:
        # Get actual device names (handle deduplication)
        if hasattr(reg, 'devices') and reg.devices:
            devices = reg.devices
        else:
            devices = [reg.device]

        # Get peripheral instances (handle deduplication)
        if hasattr(reg, 'peripheral_instances') and reg.peripheral_instances:
            periphs = reg.peripheral_instances
        else:
            periphs = [reg.peripheral]

        # Track which device has which peripherals
        for device in devices:
            # -------------------------------
            # FIX 1 (PART A): skip bad devices
            # -------------------------------
            if not device or device == "None":
                continue

            for periph in periphs:
                device_peripherals[device].add(periph)

            # Store series info if available
            if hasattr(reg, 'device_series') and reg.device_series:
                device_series[device] = reg.device_series

    # Create one summary chunk per device
    summary_chunks = []

    for device, peripherals in device_peripherals.items():
        # --------------------------------
        # FIX 1 (PART B): safety net skip
        # --------------------------------
        if not device or device == "None":
            continue

        # (rest of your existing code unchanged)
        timers = sorted([p for p in peripherals if p.startswith("TIM")])
        gpios = sorted([p for p in peripherals if p.startswith("GPIO")])
        uarts = sorted([p for p in peripherals if "UART" in p or "USART" in p])
        spis = sorted([p for p in peripherals if p.startswith("SPI")])
        i2cs = sorted([p for p in peripherals if p.startswith("I2C")])
        adcs = sorted([p for p in peripherals if p.startswith("ADC")])
        dacs = sorted([p for p in peripherals if p.startswith("DAC")])
        dmas = sorted([p for p in peripherals if p.startswith("DMA")])
        usbs = sorted([p for p in peripherals if "USB" in p])
        cans = sorted([p for p in peripherals if "CAN" in p])

        categorized = set(timers + gpios + uarts + spis + i2cs + adcs + dacs + dmas + usbs + cans)
        others = sorted([p for p in peripherals if p not in categorized])

        lines = [f"Device: {device}"]

        if device in device_series:
            lines.append(f"Series: {device_series[device]}")

        lines.append(f"\nAvailable Peripherals ({len(peripherals)} total):\n")

        if timers:
            lines.append(f"Timers ({len(timers)}): {', '.join(timers)}")
        if gpios:
            lines.append(f"GPIO ({len(gpios)}): {', '.join(gpios)}")
        if uarts:
            lines.append(f"UART/USART ({len(uarts)}): {', '.join(uarts)}")
        if spis:
            lines.append(f"SPI ({len(spis)}): {', '.join(spis)}")
        if i2cs:
            lines.append(f"I2C ({len(i2cs)}): {', '.join(i2cs)}")
        if adcs:
            lines.append(f"ADC ({len(adcs)}): {', '.join(adcs)}")
        if dacs:
            lines.append(f"DAC ({len(dacs)}): {', '.join(dacs)}")
        if dmas:
            lines.append(f"DMA ({len(dmas)}): {', '.join(dmas)}")
        if usbs:
            lines.append(f"USB ({len(usbs)}): {', '.join(usbs)}")
        if cans:
            lines.append(f"CAN ({len(cans)}): {', '.join(cans)}")

        if others:
            lines.append(f"\nSystem/Other: {', '.join(others)}")

        text = "\n".join(lines)

        chunk = TextChunk(
            id=f"device_summary_{device}",
            text=text,
            metadata={
                "type": "device_summary",
                "device": device,
                "peripheral": " ".join(sorted(list(peripherals))),  # ← ADDED THIS LINE
                "device_series": device_series.get(device, ""),
                "peripheral_count": len(peripherals),
                "peripherals": sorted(list(peripherals)),
                "timers": timers,
                "gpios": gpios,
                "uarts": uarts,
                "spis": spis,
                "i2cs": i2cs,
                "adcs": adcs,
                "dacs": dacs,
                "usbs": usbs,
                "cans": cans,
            }
        )

        summary_chunks.append(chunk)

    return summary_chunks



def create_chunks(registers: List[ParsedRegister]) -> List[TextChunk]:
    """
    Create dual-level chunks from all registers.
    
    For each peripheral, creates:
    1. ONE summary chunk (~400 chars) with all register names and config hints
    2. MULTIPLE detail chunks (~400 chars each) with full register/field details
    
    Plus one device summary chunk per device.
    
    This strategy is optimized for 512-token embedding models like all-MiniLM-L6-v2.
    
    Args:
        registers: List of parsed (and possibly deduplicated) registers
        
    Returns:
        List of all text chunks ready for embedding
    """
    # Group registers by peripheral
    peripheral_groups = _group_registers_by_peripheral(registers)
    
    print(f"\nCreating dual-level chunks (optimized for 512-token models)...")
    print(f"  Found {len(peripheral_groups)} unique peripherals")
    
    chunks = []
    summary_count = 0
    detail_count = 0
    
    for peripheral_key, periph_registers in peripheral_groups.items():
        try:
            # Create 1 summary chunk per peripheral
            summary_chunk = create_peripheral_summary_chunk(peripheral_key, periph_registers)
            chunks.append(summary_chunk)
            summary_count += 1
            
            # Create multiple detail chunks per peripheral
            detail_chunks = create_peripheral_detail_chunks(peripheral_key, periph_registers)
            chunks.extend(detail_chunks)
            detail_count += len(detail_chunks)
            
            # Show sample for first few
            if summary_count <= 3:
                print(f"  ✓ {peripheral_key}: 1 summary + {len(detail_chunks)} detail chunks ({len(periph_registers)} registers)")
        
        except Exception as e:
            print(f"  ⚠️  Failed to create chunks for {peripheral_key}: {e}")
    
    print(f"\n✓ Created {len(chunks)} peripheral chunks")
    print(f"  ({summary_count} summaries + {detail_count} details)")
    
    # Create device summary chunks
    print("\nCreating device summary chunks...")
    device_chunks = create_device_summary_chunks(registers)
    chunks.extend(device_chunks)
    print(f"✓ Created {len(device_chunks)} device summary chunks")
    
    # Final summary
    print(f"\n✓ Total: {len(chunks)} chunks")
    print(f"  - {summary_count} peripheral summaries")
    print(f"  - {detail_count} peripheral details")
    print(f"  - {len(device_chunks)} device summaries")
    
    return chunks